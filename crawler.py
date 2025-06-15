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
            '.mp3', '.mp4', '.avi', '.mov', '.webp', '.ico', '.xml', '.zip'
        ]
        if any(url.lower().endswith(ext) for ext in unwanted_extensions):
            return False

        # Patrones específicamente permitidos/valorados para turismo
        valuable_patterns = [
            '/destination', '/travel', '/tourism', '/tour', '/visit', '/vacation',
            '/holiday', '/hotel', '/accommodation', '/attractions', '/guide',
            '/places', '/things-to-do', '/city-guide', '/restaurant', '/review',
            '/experience', '/itinerary', '/trip', '/explore', '/adventure', '/hoteles',
            '/vacaciones', '/destinos', '/atracciones', '/lugares', 'cuba', 'habana',
            'varadero', '/country/', '/hotels-', 'booking.com', 'tripadvisor.com',
            'lonelyplanet.com', 'expedia.com', 'hotels.com'
        ]

        # Si la URL contiene patrones específicamente valiosos, darle prioridad
        if any(pattern in url.lower() for pattern in valuable_patterns):
            return True

        # Filtrar URLs no relevantes para turismo (más permisivo)
        unwanted_patterns = [
            '/login', '/signin', '/signup', '/register', '/account', '/cart',
            '/checkout', '/payment', '/admin', '/wp-admin', '/wp-login',
            '/careers', '/jobs', '/author/', '/tag/', '/category/technology',
            '/category/business', '/category/finance', '/forum', '/community',
            '/password', '/user', '/profile', '/settings',
            '/comment', '/feed', '/rss', '/sitemap', '/api/', '/cdn-cgi/',
            'javascript:', 'mailto:', 'tel:'
        ]

        # Rechazar URLs con patrones no deseados
        if any(pattern in url.lower() for pattern in unwanted_patterns):
            return False

        # Permitir URLs de sitios de turismo conocidos
        trusted_domains = [
            'tripadvisor.com', 'booking.com', 'expedia.com', 'hotels.com',
            'lonelyplanet.com', 'agoda.com', 'kayak.com', 'priceline.com',
            'travelocity.com', 'orbitz.com', 'frommers.com', 'roughguides.com',
            'nationalgeographic.com'
        ]
        
        if any(domain in url.lower() for domain in trusted_domains):
            return True

        return True  # Ser más permisivo por defecto

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
            # Obtenerg el título de la página
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

    def log_step(self, message: str, url: Optional[str] = None, depth: Optional[int] = None, context: str = "general"):
        """Registra y muestra un paso del crawler con contexto mejorado"""
        step = {
            "message": message,
            "url": url,
            "depth": depth,
            "context": context,
            "visited_count": len(self.visited_urls),
            "to_visit_count": len(self.urls_to_visit)
        }
        self.crawl_steps.append(step)
        
        # Formatear el mensaje según el contexto
        if context == "keyword_search":
            print(f"[Keyword Search] {message} | URL: {url} | Depth: {depth}")
        else:
            print(f"[General Crawl] {message} | URL: {url} | Depth: {depth} | Visited: {len(self.visited_urls)} | To Visit: {len(self.urls_to_visit)}")

    def google_search_links(self, keywords: list, num_results: int = 100) -> list:
        """
        Busca enlaces relevantes para las palabras clave usando múltiples estrategias.
        Devuelve una lista de URLs encontradas.
        """
        import time
        import random
        import urllib.parse

        links = []
        if not keywords:
            return links

        # Crear consulta de búsqueda
        query = ' '.join(keywords)
        encoded_query = urllib.parse.quote_plus(query)

        # Cabeceras más realistas
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }

        # Estrategia 1: DuckDuckGo
        self.log_step(f"Buscando en DuckDuckGo: '{query}'", None)
        links.extend(self._search_duckduckgo(encoded_query, headers))

        # Estrategia 2: Si DuckDuckGo no funciona, usar URLs predefinidas relevantes
        if not links:
            self.log_step("DuckDuckGo no devolvió resultados, usando URLs predefinidas relevantes", None)
            links.extend(self._get_fallback_urls(keywords))

        # Estrategia 3: Si aún no hay resultados, usar búsqueda directa en sitios conocidos
        if not links:
            self.log_step("Generando URLs directas para sitios de turismo conocidos", None)
            links.extend(self._generate_direct_urls(keywords))

        # Filtrar y validar URLs
        valid_links = []
        for link in links:
            if self.is_valid_url(link) and link not in self.visited_urls:
                valid_links.append(link)

        # Eliminar duplicados y limitar resultados
        valid_links = list(set(valid_links))[:num_results]
        
        self.log_step(f"Total de URLs válidas encontradas: {len(valid_links)}", None)
        return valid_links

    def _search_duckduckgo(self, encoded_query: str, headers: dict) -> list:
        """Busca en DuckDuckGo con múltiples intentos y selectores"""
        links = []
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    import time
                    import random
                    sleep_time = random.uniform(3, 7)
                    self.log_step(f"Esperando {sleep_time:.1f}s antes del intento {attempt+1}", None)
                    time.sleep(sleep_time)

                # Probar diferentes URLs de DuckDuckGo
                ddg_urls = [
                    f"https://duckduckgo.com/html/?q={encoded_query}",
                    f"https://duckduckgo.com/?q={encoded_query}&ia=web",
                    f"https://html.duckduckgo.com/html/?q={encoded_query}"
                ]

                for ddg_url in ddg_urls:
                    try:
                        resp = requests.get(ddg_url, headers=headers, timeout=30)
                        
                        if resp.status_code == 200:
                            soup = BeautifulSoup(resp.text, 'html.parser')
                            
                            # Múltiples selectores para diferentes versiones de DuckDuckGo
                            selectors = [
                                '.result__a',
                                'a.result__url',
                                '.web-result__title a',
                                '.result .result__title a',
                                'h2.result__title a',
                                'a[data-testid="result-title-a"]',
                                '.results .result a[href]'
                            ]
                            
                            for selector in selectors:
                                results = soup.select(selector)
                                for result in results:
                                    href = result.get('href')
                                    if href and self._is_external_url(href):
                                        links.append(href)
                                
                                if links:
                                    break
                            
                            if links:
                                break
                                
                    except Exception as e:
                        self.log_step(f"Error con URL {ddg_url}: {str(e)}", None)
                        continue

                if links:
                    break

            except Exception as e:
                self.log_step(f"Error en intento {attempt+1}: {str(e)}", None)

        return links

    def _is_external_url(self, url: str) -> bool:
        """Verifica si una URL es externa (no de DuckDuckGo)"""
        if not url:
            return False
        
        excluded_domains = [
            'duckduckgo.com',
            'duck.co',
            'duckduckgo.co',
            'javascript:',
            'mailto:',
            '#'
        ]
        
        url_lower = url.lower()
        return not any(domain in url_lower for domain in excluded_domains)

    def _get_fallback_urls(self, keywords: list) -> list:
        """Devuelve URLs predefinidas relevantes basadas en las palabras clave"""
        fallback_urls = []
        
        # Mapeo de palabras clave a URLs específicas
        keyword_mappings = {
            'cuba': [
                'https://www.tripadvisor.com/Tourism-g147270-Cuba-Vacations.html',
                'https://www.lonelyplanet.com/cuba',
                'https://www.booking.com/country/cu.html',
                'https://www.expedia.com/Cuba.d178293.Destination-Travel-Guides',
                'https://www.frommers.com/destinations/cuba'
            ],
            'hoteles': [
                'https://www.booking.com/',
                'https://www.hotels.com/',
                'https://www.expedia.com/Hotels',
                'https://www.tripadvisor.com/Hotels',
                'https://www.agoda.com/'
            ],
            'turismo': [
                'https://www.tripadvisor.com/',
                'https://www.lonelyplanet.com/',
                'https://www.nationalgeographic.com/travel/',
                'https://www.roughguides.com/'
            ]
        }
        
        # Buscar URLs relevantes basadas en las palabras clave
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for key, urls in keyword_mappings.items():
                if key in keyword_lower or keyword_lower in key:
                    fallback_urls.extend(urls)
        
        return fallback_urls

    def _generate_direct_urls(self, keywords: list) -> list:
        """Genera URLs directas para sitios conocidos de turismo"""
        direct_urls = []
        
        # Sitios base conocidos
        base_sites = [
            'https://www.tripadvisor.com/Tourism',
            'https://www.booking.com/searchresults.html',
            'https://www.lonelyplanet.com/search',
            'https://www.expedia.com/Hotel-Search'
        ]
        
        # Si las palabras clave incluyen destinos específicos, crear URLs más específicas
        if any(keyword.lower() in ['cuba', 'habana', 'varadero'] for keyword in keywords):
            direct_urls.extend([
                'https://www.tripadvisor.com/Tourism-g147270-Cuba-Vacations.html',
                'https://www.lonelyplanet.com/cuba',
                'https://www.booking.com/country/cu.html',
                'https://www.hotels.com/de1426068/hotels-cuba/',
                'https://www.expedia.com/Cuba.d178293.Destination-Travel-Guides'
            ])
        
        return direct_urls

    def crawl_from_links(self, links: list, max_depth: int = 2):
        """
        Realiza crawling recursivo de las URLs proporcionadas siguiendo enlaces internos.
        Similar al crawling general pero enfocado en URLs específicas encontradas por keywords.
        
        Args:
            links: Lista de URLs iniciales para procesar
            max_depth: Profundidad máxima de crawling recursivo (default: 3)
        
        Returns:
            int: Número de páginas procesadas
        """
        if not links:
            self.log_step("No hay nuevas URLs para procesar", None, context="keyword_search")
            return 0

        # Tomar solo las primeras 5 URLs para crawling recursivo
        initial_links = links[:5]
        self.log_step(f"Iniciando crawling recursivo de {len(initial_links)} URLs principales (profundidad máxima: {max_depth})", None, context="keyword_search")
        
        # Estructuras para el crawling recursivo específico
        keyword_visited = set()
        keyword_to_visit = set(initial_links)
        keyword_url_depths = {url: 0 for url in initial_links}
        pages_processed = 0
        
        # Límite de páginas para evitar crawling excesivo
        max_pages_per_search = 50
        
        while keyword_to_visit and pages_processed < max_pages_per_search:
            # Obtener la siguiente URL a visitar
            current_url = keyword_to_visit.pop()
            current_depth = keyword_url_depths.get(current_url, 0)
            
            # Verificar si ya visitamos esta URL (en cualquier contexto)
            if current_url in keyword_visited or current_url in self.visited_urls:
                continue
                
            # Verificar si excedimos la profundidad máxima
            if current_depth > max_depth:
                continue
            
            self.log_step(f"Procesando URL recursiva {pages_processed+1}/{max_pages_per_search}", current_url, current_depth, context="keyword_search")
            
            try:
                # Realizar la solicitud HTTP
                response = requests.get(current_url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })

                # Verificar si la solicitud fue exitosa
                if response.status_code != 200:
                    self.log_step(f"Error HTTP {response.status_code}", current_url, current_depth, context="keyword_search")
                    keyword_visited.add(current_url)
                    continue

                # Parsear el contenido HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extraer contenido relevante
                content_data = self.extract_content(current_url, soup)
                if content_data:
                    # Generar un ID único para el documento
                    doc_id = f"kw_recursive_{hash(current_url) % 10000000}_{current_depth}"

                    # Añadir a ChromaDB
                    self.collection.add(
                        documents=[content_data["content"]],
                        metadatas=[{
                            "url": content_data["url"],
                            "title": content_data["title"],
                            "source": "keyword_recursive_search",
                            "depth": current_depth,
                            "query_date": "auto_updated"
                        }],
                        ids=[doc_id]
                    )
                    self.log_step(f"Contenido añadido: {content_data['title'][:50]}...", current_url, current_depth, context="keyword_search")
                    pages_processed += 1
                else:
                    self.log_step(f"Contenido insuficiente o no relevante", current_url, current_depth, context="keyword_search")

                # Extraer y procesar enlaces si no estamos en la profundidad máxima
                if current_depth < max_depth:
                    new_links = self.get_links(current_url, soup)
                    
                    # Filtrar enlaces para mantener relevancia
                    relevant_links = self._filter_relevant_links(new_links, current_url)
                    
                    # Añadir nuevos enlaces a la cola de visitas
                    links_added = 0
                    for link in relevant_links:
                        if (link not in keyword_visited and 
                            link not in self.visited_urls and 
                            link not in keyword_to_visit and
                            links_added < 10):  # Limitar enlaces por página
                            
                            keyword_to_visit.add(link)
                            keyword_url_depths[link] = current_depth + 1
                            links_added += 1
                    
                    if links_added > 0:
                        self.log_step(f"Añadidos {links_added} enlaces para profundidad {current_depth + 1}", current_url, current_depth, context="keyword_search")

                # Marcar como visitada
                keyword_visited.add(current_url)
                self.visited_urls.add(current_url)  # También marcar en el conjunto general

            except Exception as e:
                self.log_step(f"Error procesando URL: {str(e)}", current_url, current_depth, context="keyword_search")
                keyword_visited.add(current_url)

        # Mostrar estadísticas finales
        self.log_step(f"Crawling recursivo finalizado. {pages_processed} páginas procesadas, {len(keyword_visited)} URLs visitadas", None, context="keyword_search")
        return pages_processed

    def _filter_relevant_links(self, links: list, parent_url: str) -> list:
        """
        Filtra enlaces para mantener solo los más relevantes durante el crawling recursivo.
        
        Args:
            links: Lista de enlaces encontrados
            parent_url: URL padre de donde provienen los enlaces
            
        Returns:
            list: Enlaces filtrados y priorizados
        """
        if not links:
            return []
        
        from urllib.parse import urlparse
        parent_domain = urlparse(parent_url).netloc.lower()
        
        # Categorizar enlaces por relevancia
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for link in links:
            if not self.is_valid_url(link):
                continue
                
            link_lower = link.lower()
            link_domain = urlparse(link).netloc.lower()
            
            # Alta prioridad: mismo dominio + patrones muy relevantes
            if (link_domain == parent_domain and 
                any(pattern in link_lower for pattern in [
                    '/hotel', '/accommodation', '/stay', '/resort',
                    '/destination', '/place', '/city', '/country',
                    '/tour', '/attraction', '/guide', '/review',
                    '/cuba', '/habana', '/varadero', '/santiago'
                ])):
                high_priority.append(link)
            
            # Prioridad media: mismo dominio o dominios confiables
            elif (link_domain == parent_domain or 
                  any(domain in link_domain for domain in [
                      'tripadvisor', 'booking', 'expedia', 'hotels',
                      'lonelyplanet', 'frommers', 'roughguides'
                  ])):
                medium_priority.append(link)
            
            # Baja prioridad: otros enlaces válidos
            else:
                low_priority.append(link)
        
        # Combinar con límites por categoría
        result = []
        result.extend(high_priority[:15])      # Máximo 15 de alta prioridad
        result.extend(medium_priority[:10])    # Máximo 10 de prioridad media
        result.extend(low_priority[:5])        # Máximo 5 de baja prioridad
        
        return result[:20]  # Límite total de 20 enlaces por página

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
