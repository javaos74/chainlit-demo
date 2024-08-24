from pysqlite3 import dbapi2 as sqlite3

class AuthDB:
    '''Authentication database '''
    def __init__(self):
        self.conn = sqlite3.connect("./authdb.db")
    
    def authenticate ( self, userid, passwd):
        cur = self.conn.cursor()
        cur.execute("select * from authdb where userid=? and password=?", (userid, passwd))
        result = cur.fetchone()
        return result
    

    
if __name__  == '__main__':
    db = AuthDB()
    print( db.authenticate( "charles", "1234!"))