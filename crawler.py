import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.utils import embedding_functions
import re
from typing import List, Dict, Optional
import tiktoken  # Para contar tokens


class TourismCrawler:
    def __init__(self, starting_urls: List[str], chroma_collection_name: str = "tourism_data", max_pages: int = 200, max_depth: int = 3):
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
        self.max_pages = max_pages
        self.max_depth = max_depth

        self.crawl_steps = []  # Historial de pasos del crawler

    def is_valid_url(self, url: str) -> bool:
        """
        Determina si una URL es válida para el crawler de turismo.
        Filtra URLs no relevantes como páginas de login, secciones about, etc.
        """
        # Validar formato de URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
        except:
            return False

        # Filtrar URLs de recursos estáticos
        unwanted_extensions = [
            '.css', '.js', '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.webp', '.ico', '.xml'
        ]
        if any(url.lower().endswith(ext) for ext in unwanted_extensions):
            return False

        # Filtrar URLs no relevantes para turismo
        unwanted_patterns = [
            '/login', '/signin', '/signup', '/register', '/account', '/cart',
            '/checkout', '/payment', '/admin', '/wp-admin', '/wp-login',
            '/contact', '/legal', '/terms', '/privacy', '/cookies',
            '/careers', '/jobs', '/author/', '/tag/', '/category/technology',
            '/category/business', '/category/finance', '/forum', '/community',
            '/search', '/password', '/user', '/profile', '/settings',
            '/comment', '/feed', '/rss', '/sitemap', '/api/', '/cdn-cgi/'
        ]

        # Patrones específicamente permitidos/valorados para turismo
        valuable_patterns = [
            '/destination', '/travel', '/tourism', '/tour', '/visit', '/vacation',
            '/holiday', '/hotel', '/accommodation', '/attractions', '/guide',
            '/places', '/things-to-do', '/city-guide', '/restaurant', '/review',
            '/experience', '/itinerary', '/trip', '/explore', '/adventure'
        ]

        # Si la URL contiene patrones específicamente valiosos, darle prioridad
        if any(pattern in url.lower() for pattern in valuable_patterns):
            return True

        # Rechazar URLs con patrones no deseados
        if any(pattern in url.lower() for pattern in unwanted_patterns):
            return False

        return True

    def get_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """
        Extrae enlaces de la página y los filtra para obtener solo URLs relevantes de turismo
        """
        links = []
        base_url = url

        # Extraer todos los enlaces de la página
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')

            # Ignorar anclas y URLs vacías
            if not href or href.startswith('#') or href == '/' or href == '':
                continue

            # Convertir URLs relativas a absolutas
            absolute_url = urljoin(base_url, href)

            # Validar y añadir a la lista
            if self.is_valid_url(absolute_url):
                links.append(absolute_url)

        return list(set(links))  # Eliminar duplicados

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
        """
        Extrae el contenido relevante de turismo de una página web.
        Intenta obtener el título y el contenido principal.
        """
        try:
            # Obtener el título de la página
            title = soup.title.string if soup.title else ""
            title = self.clean_text(title)

            # Eliminar elementos que normalmente no contienen contenido útil
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()

            # Buscar el contenido principal - primero intentamos con artículos o secciones principales
            content_candidates = []

            # Patrones de clases o IDs que suelen contener contenido principal en sitios de turismo
            main_content_patterns = [
                'article', 'main', 'content', 'post', 'entry', 'blog-post',
                'description', 'destination', 'attraction', 'place', 'tour-details',
                'trip-details', 'hotel-description', 'review-content'
            ]

            # Buscar por etiquetas semánticas
            for tag in ['article', 'main', 'section']:
                elements = soup.find_all(tag)
                content_candidates.extend(elements)

            # Buscar por clases o IDs que suelen contener contenido principal
            for pattern in main_content_patterns:
                # Buscar por ID
                element = soup.find(id=lambda x: x and pattern in x.lower())
                if element:
                    content_candidates.append(element)

                # Buscar por clase
                elements = soup.find_all(class_=lambda x: x and pattern in x.lower())
                content_candidates.extend(elements)

            # Si no encontramos contenido específico, usamos todos los párrafos
            if not content_candidates:
                content_candidates = soup.find_all('p')

            # Extraer el texto de los candidatos
            content_text = ""
            for candidate in content_candidates:
                # Para cada candidato, extraemos el texto
                text = candidate.get_text(separator=' ', strip=True)
                if text and len(text) > 100:  # Solo texto de cierta longitud
                    content_text += text + " "

            # Limpiar el texto
            content_text = self.clean_text(content_text)

            # Verificar si tenemos suficiente contenido
            if not title or not content_text or len(content_text) < 200:
                self.log_step("Contenido insuficiente o no relevante", url)
                return None

            # Limitar el tamaño del contenido para vectorizar (para evitar textos demasiado grandes)
            max_tokens = 1500  # Ajustar según necesidades
            if self.count_tokens(content_text) > max_tokens:
                # Truncar el texto para que quepa en los límites
                tokens = content_text.split()
                content_text = ' '.join(tokens[:max_tokens*2])  # Aproximación

            # Retornar el resultado
            return {
                "url": url,
                "title": title,
                "content": content_text
            }

        except Exception as e:
            self.log_step(f"Error extrayendo contenido: {str(e)}", url)
            return None

    def log_step(self, message: str, url: Optional[str] = None, depth: Optional[int] = None):
        """Registra y muestra un paso del crawler"""
        step = {
            "message": message,
            "url": url,
            "depth": depth,
            "visited_count": len(self.visited_urls),
            "to_visit_count": len(self.urls_to_visit)
        }
        self.crawl_steps.append(step)
        print(f"[Step] {message} | URL: {url} | Depth: {depth} | Visited: {len(self.visited_urls)} | To Visit: {len(self.urls_to_visit)}")

    def run_crawler(self):
        """
        Ejecuta el crawler para recolectar información de turismo y almacenarla en ChromaDB
        """
        page_count = 0
        url_depths = {url: 0 for url in self.starting_urls}  # Seguimiento de la profundidad de cada URL

        self.log_step("Iniciando crawler de turismo", None)

        while self.urls_to_visit and page_count < self.max_pages:
            # Obtener la siguiente URL a visitar
            url = self.urls_to_visit.pop()
            current_depth = url_depths.get(url, 0)

            # Verificar si ya visitamos esta URL
            if url in self.visited_urls:
                continue

            # Verificar si excedimos la profundidad máxima
            if current_depth > self.max_depth:
                continue

            self.log_step(f"Visitando página {page_count+1}/{self.max_pages}", url, current_depth)

            try:
                # Realizar la solicitud HTTP
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })

                # Verificar si la solicitud fue exitosa
                if response.status_code != 200:
                    self.log_step(f"Error: Código de estado HTTP {response.status_code}", url)
                    self.visited_urls.add(url)
                    continue

                # Parsear el contenido HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extraer contenido
                content_data = self.extract_content(url, soup)
                if content_data:
                    # Añadir a ChromaDB
                    self.collection.add(
                        documents=[content_data["content"]],
                        metadatas=[{
                            "url": content_data["url"],
                            "title": content_data["title"],
                            "source": "tourism_crawler"
                        }],
                        ids=[f"doc_{len(self.visited_urls)}"]
                    )
                    self.log_step(f"Contenido añadido a ChromaDB: {content_data['title']}", url)

                # Extraer y procesar enlaces si no estamos en la profundidad máxima
                if current_depth < self.max_depth:
                    new_links = self.get_links(url, soup)

                    # Añadir nuevos enlaces a la cola de visitas
                    for link in new_links:
                        if link not in self.visited_urls and link not in self.urls_to_visit:
                            self.urls_to_visit.add(link)
                            url_depths[link] = current_depth + 1

                # Marcar como visitada
                self.visited_urls.add(url)
                page_count += 1

            except Exception as e:
                self.log_step(f"Error procesando URL: {str(e)}", url)
                self.visited_urls.add(url)

        self.log_step(f"Crawler finalizado. Páginas procesadas: {page_count}", None)
        return page_count
