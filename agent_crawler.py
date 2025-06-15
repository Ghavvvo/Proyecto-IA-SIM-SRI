from autogen import Agent
from crawler import TourismCrawler

class CrawlerAgent(Agent):
    def __init__(self, name, starting_urls, max_pages=200, max_depth=2):
        super().__init__(name)
        self.crawler = TourismCrawler(starting_urls, max_pages=max_pages, max_depth=max_depth)

    def receive(self, message, sender):
        if message['type'] == 'crawl':
            # Ejecutar el crawler completo y devolver la colección
            self.crawler.run_crawler()
            return {'type': 'crawled', 'collection': self.crawler.collection}
        elif message['type'] == 'crawl_keywords':
            # Extraer palabras clave del mensaje
            keywords = message.get('keywords', [])
            if not keywords:
                return {'type': 'error', 'msg': 'No se proporcionaron palabras clave para la búsqueda'}

            # Buscar enlaces en Google basados en las palabras clave
            links = self.crawler.google_search_links(keywords)

            if not links:
                return {'type': 'error', 'msg': 'No se encontraron URLs relevantes para las palabras clave proporcionadas'}

            # Realizar crawling con profundidad 1 en los enlaces encontrados
            pages_processed = self.crawler.crawl_from_links(links, max_depth=5)

            if pages_processed > 0:
                return {'type': 'crawled', 'collection': self.crawler.collection, 'pages_processed': pages_processed}
            else:
                return {'type': 'error', 'msg': 'No se pudo actualizar la base de datos con nueva información'}

        return {'type': 'error', 'msg': 'Tipo de mensaje desconocido'}
