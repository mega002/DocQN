######################################################################
# Evidence RPC Server
#
# Running on top of RabbitMQ
#
######################################################################

###################################
# Imports
#

import pika
import pickle
from tqdm import tqdm
import simplejson as json
from anytree.exporter import JsonExporter


###################################
# Globals
#

evidence_dict_path = '../data/evidence_dict_cased.nop.pkl'
predictions_path = '../data/all_qa_para_preds.nop.json'

server_evidence_dict = None
server_prediction_dict = None


###################################
# Functions
#

def load_serialize_evidence_dict():
    global server_evidence_dict
    print("Loading evidence dict...", end='')
    with open(evidence_dict_path, 'rb') as fd:
        evidence_dict = pickle.load(fd)
    print("{} evidences".format(len(evidence_dict)))

    print("Serializing evidence dict...")
    exporter = JsonExporter()
    for key in tqdm(evidence_dict):
        evidence = evidence_dict[key]
        evidence['tree'] = exporter.export(evidence['tree'])
        evidence_dict[key] = json.dumps(evidence)

    server_evidence_dict = evidence_dict


def load_prediction_dict():
    global server_prediction_dict

    print("Loading prediction dict...", end='')
    with open(predictions_path, 'r') as fd:
        prediction_dict = json.load(fd)
    print("{} qid-eidx predictions".format(len(prediction_dict)))

    server_prediction_dict = prediction_dict


def get_evidence(evidence_title):
    return server_evidence_dict[evidence_title]


def get_response(request):
    qid, evidence_title = request.split('--', 1)
    evidence = get_evidence(evidence_title)
    if qid == '<EMPTY>':
        return evidence
    predictions = server_prediction_dict[request]

    return json.dumps([evidence, predictions])


def on_request(ch, method, props, body):
    body_str = body.decode("utf-8")
    response = get_response(body_str)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=response)
    ch.basic_ack(delivery_tag=method.delivery_tag)


###################################
# Main
#

def main():
    print("\nInitializing evidence RPC server...")
    load_serialize_evidence_dict()
    load_prediction_dict()

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))

    channel = connection.channel()
    channel.queue_declare(queue='rpc_queue')
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_request, queue='rpc_queue')

    print("\nReady for RPC requests...")
    channel.start_consuming()


if __name__ == "__main__":
    main()

