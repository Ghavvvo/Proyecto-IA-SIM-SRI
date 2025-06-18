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
                                
                                # IMPRIMIR INFORMACIÓN DEL CHUNK
                                print(f"\n📝 GUARDANDO CHUNK EN CHROMADB:")
                                print(f"   📌 ID: {doc_id}")
                                print(f"   🔗 URL: {content_data['url']}")
                                print(f"   📄 Título: {content_data['title']}...")
                                print(f"   📏 Tamaño del texto: {len(structured_text)} caracteres")
                                print(f"   🏷️ Procesado por: Gemini")
                                if 'paises' in metadata:
                                    print(f"   🌍 Países: {metadata['paises']}")
                                print(f"   📊 Metadata: {len(metadata)} campos")
                                print(f"   ✅ Chunk guardado exitosamente\n")
                                
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

    def google_search_links(self, keywords: list, num_results: int = 50, improved_query: str = None) -> list:
        """
        Busca enlaces relevantes usando múltiples motores de búsqueda.
        
        Args:
            keywords: Lista de palabras clave para la búsqueda
            num_results: Número de resultados deseados
            improved_query: Consulta mejorada por el agente de contexto (opcional)
        """
        if not keywords and not improved_query:
            return []

        # Si hay una consulta mejorada, usarla preferentemente
        if improved_query:
            print(f"🔍 Búsqueda con consulta mejorada: '{improved_query}'")
            # Convertir la consulta mejorada en lista para compatibilidad
            search_keywords = [improved_query]
        else:
            print(f"🔍 Búsqueda para palabras clave del usuario: {keywords}")
            search_keywords = keywords
        
        # Intentar con diferentes motores de búsqueda
        search_engines = [
            ("DuckDuckGo", self._search_duckduckgo_links),
            ("Bing", self._search_bing_links),
            ("Searx", self._search_searx_links),
            ("Búsqueda directa", self._fallback_direct_search)
        ]
        
        for engine_name, search_function in search_engines:
            print(f"\n🔎 Intentando con {engine_name}...")
            try:
                web_links = search_function(search_keywords, num_results_per_query=num_results)
                if web_links:
                    print(f"🌐 Encontradas {len(web_links)} URLs via {engine_name}")
                    return web_links[:num_results]
                else:
                    print(f"⚠️ {engine_name}: No se encontraron resultados")
            except Exception as e:
                print(f"❌ Error con {engine_name}: {e}")
                continue
        
        # Si ningún motor funciona
        print("\n❌ No se pudieron obtener resultados de ningún motor de búsqueda")
        print(f"   - Palabras clave utilizadas: {search_keywords}")
        print("   - Todos los motores de búsqueda fallaron o están bloqueados")
        return []
    
    def _direct_keyword_search(self, keywords: list) -> list:
        """
        Búsqueda directa construyendo URLs basadas en palabras clave.
        """
        direct_urls = []
        
        # Plantillas de URLs para búsqueda directa
        url_templates = [
            "https://www.tripadvisor.com/Search?q={keyword}",
            "https://www.booking.com/searchresults.html?ss={keyword}",
            "https://www.expedia.com/Hotel-Search?destination={keyword}",
            "https://www.hotels.com/search.do?q={keyword}",
            "https://www.lonelyplanet.com/search?q={keyword}",
            "https://www.viator.com/searchResults/all?text={keyword}",
            "https://www.getyourguide.com/s/?q={keyword}",
            "https://www.airbnb.com/s/{keyword}/homes"
        ]
        
        # Para cada palabra clave, generar URLs directas
        for keyword in keywords[:3]:  # Limitar a 3 keywords
            keyword_encoded = keyword.replace(' ', '+')
            
            for template in url_templates:
                try:
                    url = template.format(keyword=keyword_encoded)
                    direct_urls.append(url)
                except:
                    continue
        
        # Agregar algunas URLs de búsqueda específicas según el tipo de keyword
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Si es un país o ciudad, agregar URLs específicas
            if any(term in keyword_lower for term in ['cuba', 'habana', 'havana', 'varadero']):
                direct_urls.extend([
                    "https://www.tripadvisor.com/Tourism-g147270-Cuba-Vacations.html",
                    "https://www.lonelyplanet.com/cuba",
                    "https://www.booking.com/country/cu.html"
                ])
            elif any(term in keyword_lower for term in ['panama', 'panamá']):
                direct_urls.extend([
                    "https://www.tripadvisor.com/Tourism-g294479-Panama-Vacations.html",
                    "https://www.lonelyplanet.com/panama",
                    "https://www.booking.com/country/pa.html"
                ])
            elif any(term in keyword_lower for term in ['angola', 'luanda']):
                direct_urls.extend([
                    "https://www.tripadvisor.com/Tourism-g293819-Angola-Vacations.html",
                    "https://www.lonelyplanet.com/angola",
                    "https://www.booking.com/country/ao.html"
                ])
        
        return list(set(direct_urls))  # Eliminar duplicados
    
    def _search_duckduckgo_links(self, keywords: list, num_results_per_query: int = 8) -> list:
        """
        Realiza búsquedas usando la librería duckduckgo_search que maneja mejor los CAPTCHAs.
        """
        all_urls = set()
        
        try:
            # Intentar usar la librería duckduckgo_search si está disponible
            from duckduckgo_search import DDGS
            
            print("  🦆 Usando DuckDuckGo Search API...")
            
            # Crear instancia del buscador
            with DDGS() as ddgs:
                # Buscar con todas las palabras clave juntas
                query = ' '.join(keywords)
                print(f"  🔍 Buscando: '{query}'")
                
                try:
                    # Realizar búsqueda
                    results = list(ddgs.text(
                        query, 
                        max_results=num_results_per_query,
                        safesearch='off',
                        region='wt-wt'  # Mundial
                    ))
                    
                    # Extraer URLs de los resultados
                    for result in results:
                        if 'href' in result:
                            all_urls.add(result['href'])
                    
                    print(f"    ✓ Encontrados {len(all_urls)} resultados")
                    
                except Exception as e:
                    print(f"    ❌ Error en búsqueda: {e}")
            
            return list(all_urls)
            
        except ImportError:
            print("  ⚠️ Librería duckduckgo_search no disponible, usando método alternativo...")
            # Fallback al método anterior mejorado
            return self._search_with_requests(keywords, num_results_per_query)
    
    def _search_with_requests(self, keywords: list, num_results_per_query: int = 8) -> list:
        """
        Método de búsqueda alternativo usando requests directamente.
        """
        import random
        from urllib.parse import quote_plus
        
        all_urls = set()
        
        # Usar directamente las palabras clave del usuario
        search_queries = [' '.join(keywords)]  # Búsqueda con todas las palabras
        # También buscar cada palabra individualmente si hay múltiples
        if len(keywords) > 1:
            search_queries.extend(keywords[:2])  # Solo las primeras 2 palabras individuales
        
        # Configurar sesión para búsquedas
        search_session = requests.Session()
        search_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        for query in search_queries[:5]:  # Limitar consultas
            try:
                print(f"  🔍 Buscando: '{query}'")
                
                # Preparar URL de búsqueda
                encoded_query = quote_plus(query)
                search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
                
                # Realizar búsqueda con timeout más largo
                response = search_session.get(search_url, timeout=30, allow_redirects=True)
                
                # Manejar diferentes códigos de respuesta
                if response.status_code in [200, 202, 301, 302]:
                    # Si es una redirección, seguirla
                    if response.status_code in [301, 302] and 'location' in response.headers:
                        redirect_url = response.headers['location']
                        response = search_session.get(redirect_url, timeout=30)
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Debug: ver qué contiene la respuesta
                    if len(response.text) < 1000:
                        print(f"    ⚠️ Respuesta muy corta: {len(response.text)} caracteres")
                    
                    # Buscar específicamente los resultados de DuckDuckGo
                    results_found = 0
                    
                    # Método 1: Buscar enlaces en divs con clase 'result'
                    result_divs = soup.find_all('div', class_='result')
                    print(f"    📦 Encontrados {len(result_divs)} divs de resultados")
                    
                    for div in result_divs:
                        # Buscar el enlace principal del resultado
                        link = div.find('a', class_='result__a')
                        if link and link.get('href'):
                            href = link['href']
                            # Procesar URL si es necesario
                            if 'duckduckgo.com/l/?uddg=' in href:
                                try:
                                    from urllib.parse import parse_qs, urlparse, unquote
                                    params = parse_qs(urlparse(href).query)
                                    if 'uddg' in params:
                                        href = unquote(params['uddg'][0])
                                except:
                                    continue
                            
                            if href and href.startswith('http'):
                                all_urls.add(href)
                                results_found += 1
                    
                    # Método 2: Si no hay resultados, buscar todos los enlaces
                    if results_found == 0:
                        all_links = soup.find_all('a', href=True)
                        print(f"    🔗 Total de enlaces en la página: {len(all_links)}")
                    
                    for link in all_links:
                        href = link.get('href', '')
                        
                        # Procesar URLs de redirección de DuckDuckGo
                        if 'duckduckgo.com/l/?uddg=' in href:
                            try:
                                from urllib.parse import parse_qs, urlparse, unquote
                                parsed = urlparse(href)
                                params = parse_qs(parsed.query)
                                if 'uddg' in params:
                                    href = unquote(params['uddg'][0])
                            except:
                                continue
                        
                        # Verificar si es una URL válida
                        if (href and 
                            href.startswith('http') and 
                            'duckduckgo.com' not in href and
                            len(href) > 15):
                            
                            # No filtrar por relevancia turística, dejar que el usuario decida
                            all_urls.add(href)
                            results_found += 1
                            
                            if results_found >= num_results_per_query:
                                break
                    
                    print(f"    ✓ Encontrados {results_found} resultados")
                else:
                    print(f"    ⚠️ Respuesta HTTP {response.status_code}")
                
                # Pausa entre búsquedas
                time.sleep(random.uniform(1.5, 2.5))
                
            except Exception as e:
                print(f"    ❌ Error en consulta '{query}': {e}")
                continue
        
        return list(all_urls)
    
    def _search_bing_links(self, keywords: list, num_results_per_query: int = 10) -> list:
        """
        Realiza búsquedas en Bing con mejor extracción de resultados.
        """
        from urllib.parse import quote_plus
        import re
        
        all_urls = set()
        query = ' '.join(keywords)
        
        try:
            # URL de búsqueda de Bing
            encoded_query = quote_plus(query)
            search_url = f"https://www.bing.com/search?q={encoded_query}&count={num_results_per_query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            response = requests.get(search_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Método 1: Buscar en los resultados principales de Bing
                # Los resultados están en elementos <li> con clase "b_algo"
                for li in soup.find_all('li', class_='b_algo'):
                    h2 = li.find('h2')
                    if h2:
                        link = h2.find('a', href=True)
                        if link and link['href'].startswith('http'):
                            all_urls.add(link['href'])
                
                # Método 2: Buscar en divs con id "b_results"
                results_div = soup.find('div', id='b_results')
                if results_div:
                    for link in results_div.find_all('a', href=True):
                        href = link['href']
                        if (href.startswith('http') and 
                            'bing.com' not in href and 
                            'microsoft.com' not in href and
                            'microsofttranslator' not in href and
                            len(href) > 20):
                            all_urls.add(href)
                            if len(all_urls) >= num_results_per_query:
                                break
                
                # Método 3: Buscar con regex para URLs
                if len(all_urls) < 5:
                    # Buscar patrones de URL en el texto
                    url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
                    potential_urls = url_pattern.findall(str(soup))
                    for url in potential_urls:
                        if ('bing.com' not in url and 
                            'microsoft.com' not in url and
                            len(url) > 30 and
                            url.count('/') >= 3):  # URLs reales tienen varias barras
                            all_urls.add(url)
                            if len(all_urls) >= num_results_per_query:
                                break
                
                print(f"    ✓ Encontrados {len(all_urls)} resultados en Bing")
            else:
                print(f"    ⚠️ Bing respondió con código {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ Error en búsqueda Bing: {e}")
        
        return list(all_urls)[:num_results_per_query]
    
    def _search_searx_links(self, keywords: list, num_results_per_query: int = 10) -> list:
        """
        Realiza búsquedas usando instancias públicas de Searx.
        """
        from urllib.parse import quote_plus
        
        all_urls = set()
        query = ' '.join(keywords)
        
        # Lista de instancias públicas de Searx
        searx_instances = [
            "https://searx.be",
            "https://searx.info",
            "https://searx.xyz",
            "https://searx.ninja"
        ]
        
        for instance in searx_instances:
            try:
                encoded_query = quote_plus(query)
                search_url = f"{instance}/search?q={encoded_query}&format=json"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; TourismCrawler/1.0)',
                    'Accept': 'application/json'
                }
                
                response = requests.get(search_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    for result in results[:num_results_per_query]:
                        if 'url' in result:
                            all_urls.add(result['url'])
                    
                    if all_urls:
                        print(f"    ✓ Encontrados {len(all_urls)} resultados en {instance}")
                        break  # Si encontramos resultados, no probar más instancias
                        
            except Exception as e:
                continue  # Probar siguiente instancia
        
        if not all_urls:
            print("    ⚠️ No se encontraron resultados en ninguna instancia de Searx")
        
        return list(all_urls)
    
    def _fallback_direct_search(self, keywords: list, num_results_per_query: int = 10) -> list:
        """
        Método de respaldo que construye URLs directamente basándose en las palabras clave.
        """
        direct_urls = []
        
        # Sitios web populares de turismo
        tourism_sites = {
            'tripadvisor': 'https://www.tripadvisor.com/Search?q={}',
            'booking': 'https://www.booking.com/searchresults.html?ss={}',
            'expedia': 'https://www.expedia.com/Hotel-Search?destination={}',
            'hotels': 'https://www.hotels.com/search.do?q={}',
            'airbnb': 'https://www.airbnb.com/s/{}/homes',
            'lonelyplanet': 'https://www.lonelyplanet.com/search?q={}',
            'viator': 'https://www.viator.com/searchResults/all?text={}',
            'kayak': 'https://www.kayak.com/hotels/{}'
        }
        
        # Generar URLs para cada palabra clave
        for keyword in keywords:
            keyword_encoded = keyword.replace(' ', '+')
            for site_name, url_template in tourism_sites.items():
                try:
                    url = url_template.format(keyword_encoded)
                    direct_urls.append(url)
                except:
                    continue
        
        # Agregar algunas URLs específicas basadas en las palabras clave
        query = ' '.join(keywords).lower()
        
        # Detectar países/ciudades específicos y agregar URLs relevantes
        location_urls = {
            'panama': [
                'https://www.visitpanama.com/',
                'https://www.tripadvisor.com/Tourism-g294479-Panama-Vacations.html'
            ],
            'cuba': [
                'https://www.cuba.travel/',
                'https://www.tripadvisor.com/Tourism-g147270-Cuba-Vacations.html'
            ],
            'angola': [
                'https://www.tripadvisor.com/Tourism-g293819-Angola-Vacations.html',
                'https://www.lonelyplanet.com/angola'
            ],
            'caribbean': [
                'https://www.caribbean.com/',
                'https://www.tripadvisor.com/Tourism-g147237-Caribbean-Vacations.html'
            ],
            'havana': [
                'https://www.tripadvisor.com/Tourism-g147271-Havana_Cuba-Vacations.html',
                'https://www.lonelyplanet.com/cuba/havana'
            ]
        }
        
        for location, urls in location_urls.items():
            if location in query:
                direct_urls.extend(urls)
        
        # Eliminar duplicados y limitar resultados
        unique_urls = list(dict.fromkeys(direct_urls))  # Preservar orden
        
        print(f"    ✓ Generadas {len(unique_urls)} URLs directas")
        return unique_urls[:num_results_per_query]
    
    def _alternative_search(self, keywords: list) -> set:
        """
        Método alternativo de búsqueda usando Google Search (via SerpAPI simulado o scraping básico)
        """
        from urllib.parse import quote_plus
        alternative_urls = set()
        
        try:
            # Intentar con búsqueda en Google
            for keyword in keywords[:3]:  # Limitar a 3 keywords
                try:
                    query = f"{keyword} tourism travel guide"
                    encoded_query = quote_plus(query)
                    
                    # Usar un proxy de búsqueda o API alternativa
                    search_url = f"https://www.google.com/search?q={encoded_query}&num=10"
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',
                    }
                    
                    response = requests.get(search_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        # Extraer URLs del HTML de Google (método básico)
                        import re
                        # Buscar patrones de URL en el HTML
                        url_pattern = r'href="(https?://[^"]+)"'
                        found_urls = re.findall(url_pattern, response.text)
                        
                        for url in found_urls:
                            # Filtrar URLs de Google y otras no deseadas
                            if ('google.com' not in url and 
                                'googleusercontent' not in url and
                                self._is_tourism_relevant_url(url)):
                                alternative_urls.add(url)
                                if len(alternative_urls) >= 15:
                                    break
                    
                    time.sleep(2)  # Pausa entre búsquedas
                    
                except Exception as e:
                    print(f"    Error en búsqueda alternativa para '{keyword}': {e}")
                    continue
                    
        except Exception as e:
            print(f"  Error general en búsqueda alternativa: {e}")
        
        # Si aún no hay resultados, usar URLs predefinidas basadas en keywords
        if len(alternative_urls) < 5:
            print("  Complementando con URLs predefinidas...")
            predefined = self._get_predefined_urls(keywords)
            alternative_urls.update(predefined[:10])
        
        return alternative_urls
    
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
        Versión mejorada con criterios más flexibles.
        """
        url_lower = url.lower()
        
        # Dominios conocidos de turismo
        tourism_domains = [
            'tripadvisor', 'booking', 'expedia', 'hotels', 'airbnb',
            'lonelyplanet', 'frommers', 'roughguides', 'nationalgeographic',
            'timeout', 'viator', 'getyourguide', 'agoda', 'hostelworld',
            'kayak', 'trivago', 'travelocity', 'orbitz', 'priceline',
            'marriott', 'hilton', 'hyatt', 'ihg', 'accor'
        ]
        
        # Patrones relevantes expandidos
        tourism_patterns = [
            'tourism', 'travel', 'vacation', 'destination', 'attractions',
            'things-to-do', 'guide', 'visit', 'hotel', 'restaurant',
            'turismo', 'viaje', 'vacaciones', 'destino', 'atracciones',
            'hoteles', 'restaurante', 'hospedaje', 'alojamiento',
            'resort', 'lodge', 'inn', 'hostel', 'accommodation',
            'sightseeing', 'tour', 'excursion', 'adventure', 'explore',
            'beach', 'playa', 'museum', 'museo', 'park', 'parque'
        ]
        
        # Verificar dominios
        if any(domain in url_lower for domain in tourism_domains):
            return True
        
        # Verificar patrones
        if any(pattern in url_lower for pattern in tourism_patterns):
            return True
        
        # Verificar si contiene palabras clave de la consulta actual
        if self.current_query_keywords:
            for keyword in self.current_query_keywords:
                if keyword.lower() in url_lower:
                    return True
        
        # Ser más permisivo con URLs que parecen ser de contenido
        # (no son recursos estáticos ni páginas de sistema)
        unwanted_extensions = ['.css', '.js', '.jpg', '.jpeg', '.png', '.gif', '.pdf']
        unwanted_patterns = ['/login', '/signin', '/register', '/api/', '/cdn-cgi/']
        
        if not any(ext in url_lower for ext in unwanted_extensions) and \
           not any(pattern in url_lower for pattern in unwanted_patterns):
            # Si la URL parece ser contenido regular, darle una oportunidad
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

    def run_parallel_crawler_from_keywords(self, keywords: list, max_depth: int = 2, improved_query: str = None) -> int:
        """Ejecuta crawling paralelo basado en palabras clave."""
        print(f"🔍 Iniciando búsqueda paralela por palabras clave: {keywords}")
        if improved_query:
            print(f"🔍 Con consulta mejorada: '{improved_query}'")
        
        # Establecer las palabras clave para filtrar URLs
        self.current_query_keywords = keywords
        print(f"🎯 Filtrado de URLs activado para palabras clave: {keywords}")
        
        # Resetear estadísticas de filtrado
        self.urls_filtered_out = 0
        
        # Buscar URLs iniciales con la consulta mejorada si está disponible
        initial_urls = self.google_search_links(keywords, num_results=20, improved_query=improved_query)
        
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
            
            # IMPRIMIR INFORMACIÓN DEL CHUNK
            print(f"\n📝 GUARDANDO CHUNK EN CHROMADB:")
            print(f"   📌 ID: {doc_id}")
            print(f"   🔗 URL: {content_data['url']}")
            print(f"   📄 Título: {content_data['title']}...")
            print(f"   📏 Tamaño del texto: {len(content_data['content'])} caracteres")
            print(f"   🏷️ Procesado por: Crawler (sin Gemini)")
            print(f"   📊 Profundidad: {depth}")
            print(f"   🧵 Thread ID: {thread_id}")
            print(f"   ✅ Chunk guardado exitosamente\n")
            
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