import os, threading, time, json
from uipath_tools import uipathorchestratorapi as uip
from dotenv import load_dotenv

load_dotenv()


class UiPathQueueTracker(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        threading.Thread.__init__(self,group,target,name,args,kwargs)
        self.con = uip.UiPathConnection( os.environ['ORCHESTRATOR_URL'], 
                                        os.environ['ORG_NAME'], 
                                        '', #os.environ['USER_NAME'], 
                                        '', #os.environ['PASSWORD'] 
                                        True, # oauth call 
                                         os.environ['TENANT_NAME'], 
                                         os.environ['CLIENT_ID'], 
                                         os.environ['CLIENT_SECRET'], 
                                         os.environ['SCOPE'])
        self.qname = kwargs['name'] # ToolCallingQ for QueueName
        self.folder = kwargs['folder'] #{ 'Id': 2334, 'Name': 'Shared'}
        self.reference = kwargs['reference']
        self.qitem = kwargs['item'] # { 'name', value } SpecificContent
        
    
    def run(self):
        if True: #self._target == None:
            added = self.con.add_queue_items(self.qname, self.folder, self.reference, self.qitem)
            qstatus = self.con.get_queueitem_status(added['Id'], self.folder)
            while qstatus['Status'] not in ['Faulted', 'Successful']:
                time.sleep(2)
                qstatus = self.con.get_queueitem_status(added['Id'], self.folder)
                
            self._return = json.loads(qstatus['OutputData'].replace('\\r\\n', '\\r').replace('\\n', '').replace('\\r', '\\n'))['DynamicProperties']['result']
    
    def join(self, *args):
        threading.Thread.join(self,*args) 
        return self._return



if __name__ == '__main__':
    args = {
        'name': 'ToolCallingQ',
        'folder': { 'Id': 1493557, 'Name': 'Shared'},
        'reference': 'PostOffice',
        'item': {
            'postNum': '1111360334160'
        }
    }
    tracker = UiPathQueueTracker( kwargs=args)
    tracker.start()
    print (tracker.join())
    
