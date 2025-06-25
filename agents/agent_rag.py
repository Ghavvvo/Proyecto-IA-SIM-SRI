from autogen import Agent
from core.rag import RAGSystem

class RAGAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.rag_system = None

    def receive(self, message, sender):
        if message['type'] == 'init_collection':
            
            self.rag_system = RAGSystem(message['collection'])
            return {'type': 'ready'}
        elif message['type'] == 'query':
            if self.rag_system:
                answer = self.rag_system.rag_query(message['query'])
                return {'type': 'answer', 'answer': answer}
            else:
                return {'type': 'error', 'msg': 'RAG system not initialized'}
        return {'type': 'error', 'msg': 'Unknown message type'}

