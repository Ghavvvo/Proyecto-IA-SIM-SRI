import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.utils import embedding_functions
import re
from typing import List, Dict, Optional
import tiktoken
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import time
import queue


class TourismCrawler:
    def __init__(self, starting_urls: List[str], chroma_collection_name: str = "tourism_data", max_pages: int = 200, max_depth: int = 3, num_threads: int = 10):
        self.starting_urls = starting_urls
        self.visited_urls = set()
        self.urls_to_visit = queue.Queue()
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
        self.num_threads = num_threads

        # Inicializar URLs en la cola
        for url in starting_urls:
            self.urls_to_visit.put((url, 0))  # (url, depth)

        # Thread-safe structures
        self.visited_lock = threading.Lock()
        self.collection_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Estadísticas
        self.pages_processed = 0
        self.pages_added_to_db = 0
        self.errors_count = 0
        
        # Control de parada
        self.stop_crawling = threading.Event()

    def is_valid_url(self, url: str) -> bool:
        """Determina si una URL es válida para el crawler de turismo."""
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

        if any(pattern in url.lower() for pattern in valuable_patterns):
            return True

        # Filtrar URLs no relevantes para turismo
        unwanted_patterns = [
            '/login', '/signin', '/signup', '/register', '/account', '/cart',
            '/checkout', '/payment', '/admin', '/wp-admin', '/wp-login',
            '/careers', '/jobs', '/author/', '/tag/', '/category/technology',
            '/category/business', '/category/finance', '/forum', '/community',
            '/password', '/user', '/profile', '/settings',
            '/comment', '/feed', '/rss', '/sitemap', '/api/', '/cdn-cgi/',
            'javascript:', 'mailto:', 'tel:'
        ]

        if any(pattern in url.lower() for pattern in unwanted_patterns):
            return False

        return True

    def get_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extrae enlaces de la página y los filtra para obtener solo URLs relevantes de turismo"""
        links = []
        base_url = url

        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if not href or href.startswith('#') or href == '/' or href == '':
                continue

            absolute_url = urljoin(base_url, href)
            if self.is_valid_url(absolute_url):
                links.append(absolute_url)

        return list(set(links))

    def clean_text(self, text: str) -> str:
        """Limpia el texto extraído"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,;:áéíóúÁÉÍÓÚñÑ-]', '', text)
        return text

    def count_tokens(self, text: str) -> int:
        """Cuenta los tokens en el texto"""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def extract_content(self, url: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Extrae el contenido relevante de turismo de una página web."""
        try:
            title = soup.title.string if soup.title else ""
            title = self.clean_text(title)

            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()

            content_candidates = []
            main_content_patterns = [
                'article', 'main', 'content', 'post', 'entry', 'blog-post',
                'description', 'destination', 'attraction', 'place', 'tour-details',
                'trip-details', 'hotel-description', 'review-content'
            ]

            for tag in ['article', 'main', 'section']:
                elements = soup.find_all(tag)
                content_candidates.extend(elements)

            for pattern in main_content_patterns:
                element = soup.find(id=lambda x: x and pattern in x.lower())
                if element:
                    content_candidates.append(element)
                elements = soup.find_all(class_=lambda x: x and pattern in x.lower())
                content_candidates.extend(elements)

            if not content_candidates:
                content_candidates = soup.find_all('p')

            content_text = ""
            for candidate in content_candidates:
                text = candidate.get_text(separator=' ', strip=True)
                if text and len(text) > 100:
                    content_text += text + " "

            content_text = self.clean_text(content_text)

            if not title or not content_text or len(content_text) < 200:
                return None

            max_tokens = 1500
            if self.count_tokens(content_text) > max_tokens:
                tokens = content_text.split()
                content_text = ' '.join(tokens[:max_tokens*2])

            return {
                "url": url,
                "title": title,
                "content": content_text
            }

        except Exception as e:
            print(f"Error extrayendo contenido de {url}: {str(e)}")
            return None

    def _process_single_url(self, url_data: tuple) -> Optional[Dict]:
        """Procesa una sola URL en un hilo separado."""
        if self.stop_crawling.is_set():
            return None
            
        url, depth = url_data
        thread_id = threading.current_thread().ident
        
        # Verificar si ya visitamos esta URL (thread-safe)
        with self.visited_lock:
            if url in self.visited_urls:
                return None
            self.visited_urls.add(url)
        
        # Actualizar estadísticas
        with self.stats_lock:
            self.pages_processed += 1
            current_processed = self.pages_processed
            
            # Verificar si hemos alcanzado el límite
            if current_processed >= self.max_pages:
                self.stop_crawling.set()
                return None
        
        print(f"[Thread-{thread_id}] Procesando URL {current_processed}/{self.max_pages} (Depth: {depth}): {url[:80]}...")
        
        try:
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })

            if response.status_code != 200:
                with self.stats_lock:
                    self.errors_count += 1
                print(f"[Thread-{thread_id}] Error HTTP {response.status_code}: {url}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            content_data = self.extract_content(url, soup)
            
            if content_data:
                # Añadir a ChromaDB (thread-safe)
                with self.collection_lock:
                    doc_id = f"parallel_doc_{hash(url) % 10000000}_{depth}_{int(time.time())}"
                    self.collection.add(
                        documents=[content_data["content"]],
                        metadatas=[{
                            "url": content_data["url"],
                            "title": content_data["title"],
                            "source": "parallel_tourism_crawler",
                            "depth": depth,
                            "thread_id": str(thread_id)
                        }],
                        ids=[doc_id]
                    )
                
                with self.stats_lock:
                    self.pages_added_to_db += 1
                
                print(f"[Thread-{thread_id}] ✓ Contenido añadido: {content_data['title'][:50]}...")
                
                # Extraer enlaces si no estamos en la profundidad máxima
                new_links = []
                if depth < self.max_depth:
                    links = self.get_links(url, soup)
                    # Filtrar y limitar enlaces
                    filtered_links = links[:10]  # Máximo 10 enlaces por página
                    new_links = [(link, depth + 1) for link in filtered_links]
                
                return {
                    "url": url,
                    "title": content_data["title"],
                    "new_links": new_links,
                    "success": True
                }
            else:
                print(f"[Thread-{thread_id}] Contenido insuficiente: {url}")
                return None

        except Exception as e:
            with self.stats_lock:
                self.errors_count += 1
            print(f"[Thread-{thread_id}] Error procesando {url}: {str(e)}")
            return None

    def run_parallel_crawler(self) -> int:
        """
        Ejecuta el crawler en paralelo usando múltiples hilos.
        Versión mejorada sin timeouts problemáticos.
        """
        print(f"🚀 Iniciando crawler paralelo con {self.num_threads} hilos")
        print(f"📊 Objetivo: {self.max_pages} páginas máximo, profundidad máxima: {self.max_depth}")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            active_futures = []
            last_progress_time = start_time
            
            while self.pages_processed < self.max_pages and not self.stop_crawling.is_set():
                
                # Limpiar futures completados y procesar resultados
                completed_futures = []
                for future in active_futures:
                    if future.done():
                        completed_futures.append(future)
                        try:
                            result = future.result()
                            if result and result.get("success"):
                                # Añadir nuevos enlaces a la cola
                                for new_link_data in result.get("new_links", []):
                                    new_url, new_depth = new_link_data
                                    with self.visited_lock:
                                        if new_url not in self.visited_urls:
                                            self.urls_to_visit.put((new_url, new_depth))
                        except Exception as e:
                            print(f"Error procesando resultado: {str(e)}")
                
                # Remover futures completados
                for future in completed_futures:
                    active_futures.remove(future)
                
                # Añadir nuevos trabajos si hay espacio
                while (len(active_futures) < self.num_threads and 
                       not self.urls_to_visit.empty() and 
                       not self.stop_crawling.is_set()):
                    
                    try:
                        url_data = self.urls_to_visit.get_nowait()
                        future = executor.submit(self._process_single_url, url_data)
                        active_futures.append(future)
                    except queue.Empty:
                        break
                
                # Mostrar progreso cada 5 segundos
                current_time = time.time()
                if current_time - last_progress_time > 5.0:
                    elapsed = current_time - start_time
                    rate = self.pages_processed / elapsed if elapsed > 0 else 0
                    print(f"📈 Progreso: {self.pages_processed}/{self.max_pages} páginas "
                          f"({self.pages_added_to_db} añadidas a DB, {self.errors_count} errores) "
                          f"- {rate:.1f} páginas/seg - {len(active_futures)} hilos activos")
                    last_progress_time = current_time
                
                # Si no hay trabajos activos y la cola está vacía, salir
                if not active_futures and self.urls_to_visit.empty():
                    break
                
                # Pequeña pausa para evitar uso excesivo de CPU
                time.sleep(0.1)
                
                # Timeout de seguridad
                if time.time() - start_time > 300:  # 5 minutos máximo
                    print("⏰ Timeout de seguridad alcanzado, finalizando crawler...")
                    self.stop_crawling.set()
                    break
        
        # Estadísticas finales
        elapsed_time = time.time() - start_time
        avg_rate = self.pages_processed / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n🎉 Crawler paralelo finalizado!")
        print(f"📊 Estadísticas finales:")
        print(f"   • Páginas procesadas: {self.pages_processed}")
        print(f"   • Páginas añadidas a DB: {self.pages_added_to_db}")
        print(f"   • Errores: {self.errors_count}")
        print(f"   • URLs visitadas: {len(self.visited_urls)}")
        print(f"   • Tiempo total: {elapsed_time:.2f} segundos")
        print(f"   • Velocidad promedio: {avg_rate:.2f} páginas/segundo")
        print(f"   • Hilos utilizados: {self.num_threads}")
        
        return self.pages_added_to_db

    def google_search_links(self, keywords: list, num_results: int = 50) -> list:
        """Busca enlaces relevantes para las palabras clave usando URLs predefinidas."""
        links = []
        if not keywords:
            return links

        print(f"🔍 Buscando URLs para palabras clave: {keywords}")
        
        # Usar URLs predefinidas relevantes (más confiable que web scraping)
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
                    links.extend(urls)
        
        # Si no encontramos URLs específicas, usar URLs generales de turismo
        if not links:
            links = [
                'https://www.tripadvisor.com/',
                'https://www.booking.com/',
                'https://www.lonelyplanet.com/',
                'https://www.expedia.com/',
                'https://www.hotels.com/'
            ]
        
        # Eliminar duplicados y limitar resultados
        valid_links = list(set(links))[:num_results]
        print(f"✅ Encontradas {len(valid_links)} URLs relevantes")
        
        return valid_links

    def run_parallel_crawler_from_keywords(self, keywords: list, max_depth: int = 2) -> int:
        """Ejecuta crawling paralelo basado en palabras clave."""
        print(f"🔍 Iniciando búsqueda paralela por palabras clave: {keywords}")
        
        # Buscar URLs iniciales
        initial_urls = self.google_search_links(keywords, num_results=20)
        
        if not initial_urls:
            print("❌ No se encontraron URLs iniciales")
            return 0
        
        print(f"✅ Encontradas {len(initial_urls)} URLs iniciales")
        
        # Limpiar la cola y añadir nuevas URLs
        while not self.urls_to_visit.empty():
            try:
                self.urls_to_visit.get_nowait()
            except queue.Empty:
                break
        
        # Añadir URLs iniciales a la cola
        for url in initial_urls:
            self.urls_to_visit.put((url, 0))
        
        # Ajustar configuración para búsqueda por keywords
        original_max_depth = self.max_depth
        self.max_depth = max_depth
        
        # Ejecutar crawler paralelo
        result = self.run_parallel_crawler()
        
        # Restaurar configuración original
        self.max_depth = original_max_depth
        
        return result

    # Métodos de compatibilidad para mantener la interfaz existente
    def crawl_from_links(self, links: list, max_depth: int = 2):
        """Método de compatibilidad - redirige al crawler paralelo"""
        print("⚠️ Usando crawler paralelo en lugar del método legacy crawl_from_links")
        
        # Limpiar la cola y añadir los enlaces proporcionados
        while not self.urls_to_visit.empty():
            try:
                self.urls_to_visit.get_nowait()
            except queue.Empty:
                break
        
        # Añadir enlaces a la cola
        for url in links[:20]:  # Limitar a 20 URLs
            self.urls_to_visit.put((url, 0))
        
        # Ejecutar crawler paralelo
        return self.run_parallel_crawler()

    def run_crawler(self):
        """Método de compatibilidad - redirige al crawler paralelo"""
        print("⚠️ Usando crawler paralelo en lugar del método secuencial legacy")
        return self.run_parallel_crawler()