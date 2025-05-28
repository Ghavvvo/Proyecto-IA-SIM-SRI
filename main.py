import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.utils import embedding_functions
import re
from typing import List, Dict, Optional
import tiktoken  # Para contar tokens


class TourismCrawler:
    def __init__(self, starting_urls: List[str], chroma_collection_name: str = "tourism_data"):
        self.starting_urls = starting_urls
        self.visited_urls = set()
        self.urls_to_visit = set(starting_urls)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")

        # Usar el embedding model de Sentence Transformers
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Crear o obtener la colección
        self.collection = self.chroma_client.get_or_create_collection(
            name=chroma_collection_name,
            embedding_function=self.sentence_transformer_ef
        )

        # Configuración del crawler
        self.max_pages = 1000
        self.max_depth = 3
        self.domain_whitelist = [
            'tripadvisor', 'lonelyplanet', 'booking.com',
            'expedia', 'airbnb', 'visit', 'tourism'
        ]

    def is_valid_url(self, url: str) -> bool:
        """Verifica si la URL es válida para el crawling"""
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False

        # Verificar si el dominio está en la whitelist
        if not any(domain in parsed.netloc for domain in self.domain_whitelist):
            return False

        # Excluir ciertos tipos de archivos
        if any(parsed.path.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip']):
            return False

        return True

    def get_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extrae todos los links válidos de una página"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
        return links

    def clean_text(self, text: str) -> str:
        """Limpia el texto extraído"""
        # Eliminar múltiples espacios y saltos de línea
        text = re.sub(r'\s+', ' ', text).strip()
        # Eliminar caracteres especiales no deseados
        text = re.sub(r'[^\w\s.,;:áéíóúÁÉÍÓÚñÑ-]', '', text)
        return text

    def count_tokens(self, text: str) -> int:
        """Cuenta los tokens en el texto (para controlar el tamaño)"""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def extract_content(self, url: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Extrae el contenido relevante de la página"""
        # Eliminar elementos no deseados
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()

        # Extraer título
        title = soup.title.string if soup.title else ""

        # Extraer el texto principal (puedes ajustar esto según la estructura de los sitios)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
        if not main_content:
            main_content = soup.body

        text = self.clean_text(main_content.get_text()) if main_content else ""

        # Verificar que el contenido sea relevante (más de 100 tokens)
        if self.count_tokens(text) < 100:
            return None

        return {
            "url": url,
            "title": title,
            "content": text,
            "source": urlparse(url).netloc
        }

    def crawl_page(self, url: str, current_depth: int):
        """Procesa una página individual"""
        try:
            print(f"Crawling: {url} (Depth: {current_depth})")
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; TourismCrawler/1.0; +https://example.com/bot)'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            content = self.extract_content(url, soup)

            if content:
                # Almacenar en ChromaDB
                self.collection.add(
                    documents=[content["content"]],
                    metadatas=[{
                        "url": content["url"],
                        "title": content["title"],
                        "source": content["source"]
                    }],
                    ids=[url]  # Usamos la URL como ID único
                )
                print(f"Added content from {url} to ChromaDB")

            # Solo seguir explorando si no hemos alcanzado la profundidad máxima
            if current_depth < self.max_depth:
                links = self.get_links(url, soup)
                for link in links:
                    if link not in self.visited_urls and link not in self.urls_to_visit:
                        self.urls_to_visit.add(link)

        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
        finally:
            self.visited_urls.add(url)
            if url in self.urls_to_visit:
                self.urls_to_visit.remove(url)

    def run_crawler(self):
        """Ejecuta el crawler desde las URLs iniciales"""
        current_depth = 0
        while self.urls_to_visit and len(self.visited_urls) < self.max_pages:
            urls_at_current_depth = list(self.urls_to_visit)
            for url in urls_at_current_depth:
                if len(self.visited_urls) >= self.max_pages:
                    break
                self.crawl_page(url, current_depth)
            current_depth += 1


# Ejemplo de uso
if __name__ == "__main__":
    starting_urls = [
        "https://www.tripadvisor.com/",
        "https://www.lonelyplanet.com/",
        "https://www.booking.com/",
        "https://www.expedia.com/",
        "https://www.airbnb.com/",
        "https://www.visitacity.com/",
        "https://www.tourism.com/"
    ]

    crawler = TourismCrawler(starting_urls)
    crawler.run_crawler()