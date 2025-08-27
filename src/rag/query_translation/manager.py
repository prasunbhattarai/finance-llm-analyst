from .multi_query import multi_query

class QueryTranslation:
    def __init__(self,cfg, vectorstore= None):
        self.vectorstore = vectorstore
        self.chain = multi_query(cfg)


    
    def multiquery(self, question):
        queries= self.chain.invoke({"questin": question})
        return self._fetch_document(queries)    