from autogen import Agent, GroupChatManager
from urls import starting_urls

class CoordinatorAgent(Agent):
    def __init__(self, name, crawler_agent, rag_agent):
        super().__init__(name)
        self.crawler_agent = crawler_agent
        self.rag_agent = rag_agent

    def start(self):
        # Verificar si la colección ya tiene datos antes de correr el crawler
        if hasattr(self.crawler_agent.crawler.collection, 'count'):
            try:
                count = self.crawler_agent.crawler.collection.count()
            except Exception:
                count = 0
        else:
            # Fallback si no existe el método count
            try:
                results = self.crawler_agent.crawler.collection.query(query_texts=["test"], n_results=1)
                count = len(results.get('documents', [[]])[0])
            except Exception:
                count = 0
        if count > 0:
            # Si ya hay datos, inicializar el RAG directamente
            self.rag_agent.receive({'type': 'init_collection', 'collection': self.crawler_agent.crawler.collection}, self)
        else:
            # Si no hay datos, correr el crawler
            crawl_result = self.crawler_agent.receive({'type': 'crawl'}, self)
            if crawl_result['type'] == 'crawled':
                self.rag_agent.receive({'type': 'init_collection', 'collection': crawl_result['collection']}, self)

    def ask(self, query):
        # Consultar al agente RAG
        response = self.rag_agent.receive({'type': 'query', 'query': query}, self)
        if response['type'] == 'answer':
            return response['answer']
        return response.get('msg', 'Error')

