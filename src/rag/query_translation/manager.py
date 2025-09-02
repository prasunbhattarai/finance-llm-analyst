from .multi_query import multi_query
from .decomposition import decompose_query

class QueryTranslation:
    def __init__(self,cfg, vectorstore= None):
        self.vectorstore = vectorstore
        self.multiquery_chain = multi_query(cfg)
        self.decompostion_chain = decompose_query(cfg)



    
    def multiquery(self, question ):
        queries= self.multiquery_chain.invoke({"questin": question})
        return self._fetch_document(queries)

    def decompose(self, question):
        queries= self.decompostion_chain.invoke({"questin": question})
    
    def hyde():
        pass
    