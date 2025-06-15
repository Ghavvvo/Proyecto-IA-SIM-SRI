from autogen import Agent
from crawler import TourismCrawler

class CrawlerAgent(Agent):
    def __init__(self, name, starting_urls, max_pages=100, max_depth=2, num_threads=10):
        super().__init__(name)
        # Crear crawler con soporte para paralelismo mejorado
        self.crawler = TourismCrawler(
            starting_urls=starting_urls, 
            max_pages=max_pages, 
            max_depth=max_depth,
            num_threads=num_threads  # 10 hilos por defecto
        )

    def receive(self, message, sender):
        if message['type'] == 'crawl':
            print(f"🚀 Iniciando crawler paralelo con {self.crawler.num_threads} hilos...")
            # Ejecutar el crawler paralelo y devolver la colección
            pages_processed = self.crawler.run_parallel_crawler()
            return {
                'type': 'crawled', 
                'collection': self.crawler.collection,
                'pages_processed': pages_processed,
                'threads_used': self.crawler.num_threads
            }
            
        elif message['type'] == 'crawl_keywords':
            # Extraer palabras clave del mensaje
            keywords = message.get('keywords', [])
            if not keywords:
                return {'type': 'error', 'msg': 'No se proporcionaron palabras clave para la búsqueda'}

            print(f"🔍 Iniciando búsqueda paralela por palabras clave: {keywords}")
            print(f"⚡ Usando {self.crawler.num_threads} hilos en paralelo")
            
            # Usar el método paralelo para crawling basado en keywords
            pages_processed = self.crawler.run_parallel_crawler_from_keywords(keywords, max_depth=3)

            if pages_processed > 0:
                return {
                    'type': 'crawled', 
                    'collection': self.crawler.collection, 
                    'pages_processed': pages_processed,
                    'keywords_used': keywords,
                    'threads_used': self.crawler.num_threads
                }
            else:
                return {'type': 'error', 'msg': 'No se pudo actualizar la base de datos con nueva información'}

        return {'type': 'error', 'msg': 'Tipo de mensaje desconocido'}