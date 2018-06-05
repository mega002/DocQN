######################################################################
# Evidence RPC client
#
# Running on top of RabbitMQ
#
######################################################################

###################################
# Imports
#

import pika
import uuid
import simplejson as json
from anytree.importer import JsonImporter
import time


###################################
# Classes
#

class EvidenceRpcClient(object):
    def __init__(self, coupled=False):
        self.tree_importer = JsonImporter()
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.coupled = coupled

        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(self.on_response, no_ack=True, queue=self.callback_queue)

    def deserialize(self, response):
        res = json.loads(response)
        if self.coupled:
            res['tree'] = self.tree_importer.import_(res['tree'])
        else:
            res[0] = json.loads(res[0])
            res[0]['tree'] = self.tree_importer.import_(res[0]['tree'])

        return res

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, request):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        try:
            self.channel.basic_publish(exchange='',
                                       routing_key='rpc_queue',
                                       properties=pika.BasicProperties(
                                             reply_to=self.callback_queue,
                                             correlation_id=self.corr_id,
                                             ),
                                       body=str(request))
        except:
            time.sleep(5)
            self.channel.basic_publish(exchange='',
                                       routing_key='rpc_queue',
                                       properties=pika.BasicProperties(
                                           reply_to=self.callback_queue,
                                           correlation_id=self.corr_id,
                                       ),
                                       body=str(request))

        while self.response is None:
            self.connection.process_data_events()

        return self.deserialize(self.response)

