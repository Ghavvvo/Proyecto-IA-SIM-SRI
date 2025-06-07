from autogen import Agent
from crawler import TourismCrawler

class CrawlerAgent(Agent):
    def __init__(self, name, starting_urls, max_pages=200, max_depth=2):
        super().__init__(name)
        self.crawler = TourismCrawler(starting_urls, max_pages=max_pages, max_depth=max_depth)

    def receive(self, message, sender):
        if message['type'] == 'crawl':
            # Ejecutar el crawler y devolver la colección
            self.crawler.run_crawler()
            return {'type': 'crawled', 'collection': self.crawler.collection}
        return {'type': 'error', 'msg': 'Unknown message type'}

