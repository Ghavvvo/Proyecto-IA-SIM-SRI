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
import json


class TourismCrawler:
    def __init__(self, starting_urls: List[str], chroma_collection_name: str = "tourism_data", max_pages: int = 100, max_depth: int = 3, num_threads: int = 10, enable_gemini_processing: bool = True):
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
        
        # Palabras clave de la consulta actual (para filtrar URLs)
        self.current_query_keywords = []

        # Inicializar URLs en la cola
        for url in starting_urls:
            self.urls_to_visit.put((url, 0))  # (url, depth)

        # Thread-safe structures
        self.visited_lock = threading.Lock()
        self.collection_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.queue_lock = threading.Lock()  # Para controlar el acceso a la cola
        
        # Estadísticas
        self.pages_processed = 0
        self.pages_added_to_db = 0
        self.errors_count = 0
        self.urls_filtered_out = 0  # URLs filtradas por no tener palabras en común
        
        # Control de parada
        self.stop_crawling = threading.Event()
        
        # Procesamiento con Gemini
        self.enable_gemini_processing = enable_gemini_processing
        self.processor_agent = None
        if self.enable_gemini_processing:
            try:
                from agent_processor import ProcessorAgent
                self.processor_agent = ProcessorAgent()
                print("✅ Procesamiento con Gemini habilitado")
            except Exception as e:
                print(f"⚠️ No se pudo habilitar el procesamiento con Gemini: {e}")
                self.enable_gemini_processing = False
        
        # Estadísticas de procesamiento
        self.gemini_processed = 0
        self.gemini_errors = 0

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

    def _has_common_keywords(self, url: str, text_content: str = "") -> bool:
        """
        Verifica si una URL o su contenido tiene palabras en común con las palabras clave de la consulta.
        Versión mejorada con expansión de palabras clave y sinónimos.
        
        Args:
            url (str): URL a verificar
            text_content (str): Contenido de texto adicional para verificar (opcional)
            
        Returns:
            bool: True si tiene palabras en común, False en caso contrario
        """
        if not self.current_query_keywords:
            return True  # Si no hay palabras clave definidas, permitir todas las URLs
        
        # Combinar URL y contenido para la verificación
        combined_text = f"{url.lower()} {text_content.lower()}"
        
        # Expandir palabras clave con sinónimos y variaciones
        expanded_keywords = self._expand_keywords(self.current_query_keywords)
        
        # Verificar si alguna palabra clave expandida está presente
        for keyword in expanded_keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower and keyword_lower in combined_text:
                return True
        
        return False
    
    def _expand_keywords(self, keywords: List[str]) -> List[str]:
        """
        Expande las palabras clave con sinónimos y variaciones para mejorar el matching.
        """
        expanded = set(keywords)  # Incluir palabras originales
        
        # Diccionario de sinónimos y variaciones comunes
        synonyms = {
            'hoteles': ['hotel', 'hotels', 'accommodation', 'alojamiento', 'hospedaje', 'lodging', 'resort', 'inn'],
            'turismo': ['tourism', 'tourist', 'travel', 'trip', 'vacation', 'holiday', 'viaje', 'destination'],
            'restaurante': ['restaurant', 'dining', 'food', 'comida', 'gastronomia', 'cuisine'],
            'playa': ['beach', 'coast', 'coastal', 'seaside', 'shore', 'waterfront'],
            'ciudad': ['city', 'urban', 'downtown', 'centro', 'metropolitan'],
            'cultura': ['culture', 'cultural', 'heritage', 'history', 'historic', 'museum'],
            'aventura': ['adventure', 'outdoor', 'activity', 'activities', 'excursion', 'tour']
        }
        
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            
            # Añadir sinónimos directos
            if keyword_lower in synonyms:
                expanded.update(synonyms[keyword_lower])
            
            # Añadir variaciones (plural/singular)
            if keyword_lower.endswith('s') and len(keyword_lower) > 3:
                expanded.add(keyword_lower[:-1])  # Singular
            else:
                expanded.add(keyword_lower + 's')  # Plural
            
            # Añadir variaciones sin acentos
            accent_map = str.maketrans('áéíóúñü', 'aeiounn')
            no_accent = keyword_lower.translate(accent_map)
            if no_accent != keyword_lower:
                expanded.add(no_accent)
        
        return list(expanded)

    def _extract_link_text(self, a_tag) -> str:
        """
        Extrae el texto del enlace y elementos cercanos para análisis de relevancia.
        
        Args:
            a_tag: Elemento <a> de BeautifulSoup
            
        Returns:
            str: Texto combinado del enlace y contexto
        """
        link_text = a_tag.get_text(strip=True)
        
        # Obtener texto del título si existe
        title_text = a_tag.get('title', '')
        
        # Obtener texto del elemento padre para contexto adicional
        parent_text = ""
        if a_tag.parent:
            parent_text = a_tag.parent.get_text(strip=True)[:100]  # Limitar a 100 caracteres
        
        return f"{link_text} {title_text} {parent_text}"

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
                # Verificar si la URL tiene palabras en común con la consulta
                link_text = self._extract_link_text(a_tag)
                if self._has_common_keywords(absolute_url, link_text):
                    links.append(absolute_url)
                else:
                    # Incrementar contador de URLs filtradas
                    with self.stats_lock:
                        self.urls_filtered_out += 1

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

            if not title or not content_text or len(content_text) < 100:
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
                # Procesar con Gemini si está habilitado
                if self.enable_gemini_processing and self.processor_agent:
                    try:
                        # Procesar contenido con Gemini
                        processor_response = self.processor_agent.receive({
                            'type': 'process_content',
                            'content_data': content_data
                        }, self)
                        
                        if processor_response.get('success') and processor_response.get('data'):
                            processed_data = processor_response['data']
                            
                            # Guardar el contenido procesado estructurado
                            with self.collection_lock:
                                # Convertir el JSON estructurado a texto para embeddings
                                structured_text = self._format_structured_data(processed_data)
                                
                                doc_id = f"gemini_doc_{hash(url) % 10000000}_{depth}_{int(time.time())}"
                                
                                # Guardar con metadata enriquecida
                                metadata = {
                                    "url": content_data["url"],
                                    "title": content_data["title"],
                                    "source": "parallel_tourism_crawler",
                                    "depth": depth,
                                    "thread_id": str(thread_id),
                                    "processed_by_gemini": True,
                                    "structured_data": json.dumps(processed_data, ensure_ascii=False)
                                }
                                
                                # Añadir información de países si existe
                                if 'paises' in processed_data and processed_data['paises']:
                                    paises_nombres = [p.get('nombre', '') for p in processed_data['paises']]
                                    metadata['paises'] = ', '.join(paises_nombres)
                                
                                self.collection.add(
                                    documents=[structured_text],
                                    metadatas=[metadata],
                                    ids=[doc_id]
                                )
                            
                            with self.stats_lock:
                                self.pages_added_to_db += 1
                                self.gemini_processed += 1
                            
                            print(f"[Thread-{thread_id}] ✅ Contenido procesado con Gemini: {content_data['title'][:50]}...")
                        else:
                            # Si falla el procesamiento, guardar el contenido original
                            self._save_original_content(content_data, depth, thread_id)
                            
                    except Exception as e:
                        print(f"[Thread-{thread_id}] ⚠️ Error en procesamiento Gemini: {e}")
                        with self.stats_lock:
                            self.gemini_errors += 1
                        # Guardar contenido original como fallback
                        self._save_original_content(content_data, depth, thread_id)
                else:
                    # Si Gemini no está habilitado, guardar contenido original
                    self._save_original_content(content_data, depth, thread_id)
                
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
        if self.current_query_keywords:
            print(f"   • URLs filtradas por palabras clave: {self.urls_filtered_out}")
            print(f"   • Palabras clave utilizadas: {self.current_query_keywords}")
        if self.enable_gemini_processing:
            print(f"   • Páginas procesadas con Gemini: {self.gemini_processed}")
            print(f"   • Errores de procesamiento Gemini: {self.gemini_errors}")
            if self.processor_agent:
                stats = self.processor_agent.get_stats()
                print(f"   • Tasa de éxito Gemini: {stats['success_rate']:.2%}")
        print(f"   • Tiempo total: {elapsed_time:.2f} segundos")
        print(f"   • Velocidad promedio: {avg_rate:.2f} páginas/segundo")
        print(f"   • Hilos utilizados: {self.num_threads}")
        
        return self.pages_added_to_db

    def google_search_links(self, keywords: list, num_results: int = 50) -> list:
        """
        Busca enlaces relevantes usando DuckDuckGo y URLs predefinidas como fallback.
        Versión mejorada con búsqueda web real.
        """
        all_links = []
        if not keywords:
            return all_links

        print(f"🔍 Búsqueda mejorada para palabras clave: {keywords}")
        
        # 1. Intentar búsqueda web real en DuckDuckGo
        try:
            web_links = self._search_duckduckgo_links(keywords, num_results_per_query=8)
            all_links.extend(web_links)
            print(f"🌐 Encontradas {len(web_links)} URLs via DuckDuckGo")
        except Exception as e:
            print(f"⚠️ Error en búsqueda DuckDuckGo: {e}")
        
        # 2. URLs predefinidas como fallback y complemento
        predefined_links = self._get_predefined_urls(keywords)
        all_links.extend(predefined_links)
        print(f"📋 Añadidas {len(predefined_links)} URLs predefinidas")
        
        # 3. Limpiar duplicados y filtrar
        unique_links = list(set(all_links))
        valid_links = [url for url in unique_links if self.is_valid_url(url)]
        
        # Limitar resultados finales
        final_links = valid_links[:num_results]
        print(f"✅ Total de URLs únicas y válidas: {len(final_links)}")
        
        return final_links
    
    def _search_duckduckgo_links(self, keywords: list, num_results_per_query: int = 8) -> list:
        """
        Realiza búsquedas reales en DuckDuckGo.
        """
        import random
        from urllib.parse import quote_plus
        
        all_urls = set()
        
        # Generar consultas de búsqueda
        search_queries = self._generate_search_queries(keywords)
        
        # Configurar sesión para búsquedas
        search_session = requests.Session()
        search_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        for query in search_queries[:6]:  # Limitar a 6 consultas máximo
            try:
                print(f"  🔍 Buscando: '{query}'")
                
                # Preparar URL de búsqueda
                encoded_query = quote_plus(query)
                search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
                
                # Realizar búsqueda
                response = search_session.get(search_url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extraer enlaces de resultados
                    result_links = soup.find_all('a', class_='result__a')
                    
                    for link in result_links[:num_results_per_query]:
                        href = link.get('href', '')
                        if href and not href.startswith('/') and self._is_tourism_relevant_url(href):
                            all_urls.add(href)
                
                # Pausa entre búsquedas
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                print(f"    ❌ Error en consulta '{query}': {e}")
                continue
        
        return list(all_urls)
    
    def _generate_search_queries(self, keywords: list) -> list:
        """
        Genera consultas de búsqueda efectivas.
        """
        queries = []
        
        # Términos de turismo para combinar
        tourism_terms = ['tourism', 'travel', 'visit', 'guide', 'attractions', 'hotels']
        
        for keyword in keywords:
            # Consultas básicas
            queries.append(f"{keyword} tourism")
            queries.append(f"{keyword} travel guide")
            queries.append(f"visit {keyword}")
            
            # Consultas específicas según el tipo de keyword
            keyword_lower = keyword.lower()
            if any(term in keyword_lower for term in ['hotel', 'accommodation', 'hospedaje']):
                queries.append(f"best hotels {keyword}")
                queries.append(f"{keyword} booking")
            elif any(term in keyword_lower for term in ['restaurant', 'food', 'comida']):
                queries.append(f"best restaurants {keyword}")
            else:
                queries.append(f"{keyword} attractions")
        
        # Consulta combinada si hay múltiples keywords
        if len(keywords) > 1:
            combined = ' '.join(keywords[:2])  # Máximo 2 palabras
            queries.append(f"{combined} tourism")
        
        return queries
    
    def _is_tourism_relevant_url(self, url: str) -> bool:
        """
        Verifica si una URL es relevante para turismo (para búsquedas web).
        """
        url_lower = url.lower()
        
        # Dominios conocidos de turismo
        tourism_domains = [
            'tripadvisor', 'booking', 'expedia', 'hotels', 'airbnb',
            'lonelyplanet', 'frommers', 'roughguides', 'nationalgeographic',
            'timeout', 'viator', 'getyourguide', 'agoda', 'hostelworld'
        ]
        
        # Patrones relevantes
        tourism_patterns = [
            'tourism', 'travel', 'vacation', 'destination', 'attractions',
            'things-to-do', 'guide', 'visit', 'hotel', 'restaurant'
        ]
        
        # Verificar dominios
        if any(domain in url_lower for domain in tourism_domains):
            return True
        
        # Verificar patrones
        if any(pattern in url_lower for pattern in tourism_patterns):
            return True
        
        return False
    
    def _get_predefined_urls(self, keywords: list) -> list:
        """
        Obtiene URLs predefinidas como fallback y complemento.
        """
        keyword_mappings = {
            'cuba': [
                'https://www.tripadvisor.com/Tourism-g147270-Cuba-Vacations.html',
                'https://www.lonelyplanet.com/cuba',
                'https://www.booking.com/country/cu.html',
                'https://www.expedia.com/Cuba.d178293.Destination-Travel-Guides'
            ],
            'angola': [
                'https://www.lonelyplanet.com/angola',
                'https://www.tripadvisor.com/Tourism-g293819-Angola-Vacations.html',
                'https://www.booking.com/country/ao.html'
            ],
            'angoola': [  # Variación específica
                'https://www.lonelyplanet.com/angola',
                'https://www.tripadvisor.com/Tourism-g293819-Angola-Vacations.html'
            ],
            'hoteles': [
                'https://www.booking.com/',
                'https://www.hotels.com/',
                'https://www.expedia.com/Hotels',
                'https://www.tripadvisor.com/Hotels'
            ],
            'hotel': [
                'https://www.booking.com/',
                'https://www.hotels.com/',
                'https://www.agoda.com/'
            ],
            'turismo': [
                'https://www.tripadvisor.com/',
                'https://www.lonelyplanet.com/',
                'https://www.nationalgeographic.com/travel/'
            ]
        }
        
        links = []
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for key, urls in keyword_mappings.items():
                if key in keyword_lower or keyword_lower in key:
                    links.extend(urls)
        
        # URLs generales si no hay coincidencias específicas
        if not links:
            links = [
                'https://www.tripadvisor.com/',
                'https://www.booking.com/',
                'https://www.lonelyplanet.com/',
                'https://www.expedia.com/',
                'https://www.hotels.com/'
            ]
        
        return links

    def run_parallel_crawler_from_keywords(self, keywords: list, max_depth: int = 2) -> int:
        """Ejecuta crawling paralelo basado en palabras clave."""
        print(f"🔍 Iniciando búsqueda paralela por palabras clave: {keywords}")
        
        # Establecer las palabras clave para filtrar URLs
        self.current_query_keywords = keywords
        print(f"🎯 Filtrado de URLs activado para palabras clave: {keywords}")
        
        # Resetear estadísticas de filtrado
        self.urls_filtered_out = 0
        
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
    
    def _save_original_content(self, content_data: Dict, depth: int, thread_id: int):
        """Guarda el contenido original sin procesar con Gemini"""
        with self.collection_lock:
            doc_id = f"parallel_doc_{hash(content_data['url']) % 10000000}_{depth}_{int(time.time())}"
            self.collection.add(
                documents=[content_data["content"]],
                metadatas=[{
                    "url": content_data["url"],
                    "title": content_data["title"],
                    "source": "parallel_tourism_crawler",
                    "depth": depth,
                    "thread_id": str(thread_id),
                    "processed_by_gemini": False
                }],
                ids=[doc_id]
            )
        
        with self.stats_lock:
            self.pages_added_to_db += 1
        
        print(f"[Thread-{thread_id}] ✓ Contenido añadido (sin Gemini): {content_data['title'][:50]}...")
    
    def _format_structured_data(self, processed_data: Dict) -> str:
        """
        Convierte los datos estructurados en un texto formateado para embeddings
        """
        formatted_text = []
        
        # Información de la fuente
        if 'source_title' in processed_data:
            formatted_text.append(f"Título: {processed_data['source_title']}")
        
        # Procesar información por países
        if 'paises' in processed_data:
            for pais in processed_data['paises']:
                pais_text = []
                
                if 'nombre' in pais:
                    pais_text.append(f"\nPaís: {pais['nombre']}")
                
                # Hoteles
                if 'hoteles' in pais and pais['hoteles']:
                    pais_text.append("\nHoteles:")
                    for hotel in pais['hoteles']:
                        hotel_info = []
                        if 'nombre' in hotel:
                            hotel_info.append(f"- {hotel['nombre']}")
                        if 'localidad' in hotel:
                            hotel_info.append(f"en {hotel['localidad']}")
                        if 'clasificacion' in hotel:
                            hotel_info.append(f"({hotel['clasificacion']})")
                        if 'precio_promedio' in hotel:
                            hotel_info.append(f"Precio: {hotel['precio_promedio']}")
                        pais_text.append(' '.join(hotel_info))
                
                # Lugares turísticos
                if 'lugares_turisticos' in pais and pais['lugares_turisticos']:
                    pais_text.append("\nLugares turísticos:")
                    for lugar in pais['lugares_turisticos']:
                        lugar_info = []
                        if 'nombre' in lugar:
                            lugar_info.append(f"- {lugar['nombre']}")
                        if 'localidad' in lugar:
                            lugar_info.append(f"en {lugar['localidad']}")
                        if 'tipo' in lugar:
                            lugar_info.append(f"({lugar['tipo']})")
                        if 'precio_entrada' in lugar:
                            lugar_info.append(f"Entrada: {lugar['precio_entrada']}")
                        pais_text.append(' '.join(lugar_info))
                
                # Información adicional
                if 'precio_promedio_visita' in pais:
                    pais_text.append(f"\nPrecio promedio de visita: {pais['precio_promedio_visita']}")
                if 'mejor_epoca' in pais:
                    pais_text.append(f"Mejor época para visitar: {pais['mejor_epoca']}")
                if 'informacion_adicional' in pais:
                    pais_text.append(f"Información adicional: {pais['informacion_adicional']}")
                
                formatted_text.extend(pais_text)
        
        # Si no hay información de países, incluir cualquier otra información
        if not formatted_text and isinstance(processed_data, dict):
            for key, value in processed_data.items():
                if key not in ['source_url', 'source_title', 'processed_by'] and value:
                    formatted_text.append(f"{key}: {value}")
        
        return '\n'.join(formatted_text) if formatted_text else "Sin información estructurada disponible"