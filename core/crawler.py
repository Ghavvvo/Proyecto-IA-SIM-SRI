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
import os
from datetime import datetime


class TourismCrawler:
    def __init__(self, starting_urls: List[str], chroma_collection_name: str = "tourism_data", max_pages: int = 100, max_depth: int = 3, num_threads: int = 10, enable_mistral_processing: bool = True):
        self.starting_urls = starting_urls
        self.visited_urls = set()
        self.urls_to_visit = queue.Queue()
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")

        
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        
        self.collection = self.chroma_client.get_or_create_collection(
            name=chroma_collection_name,
            embedding_function=self.sentence_transformer_ef
        )

        
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.num_threads = num_threads
        
        
        self.current_query_keywords = []

        
        for url in starting_urls:
            self.urls_to_visit.put((url, 0))  

        
        self.visited_lock = threading.Lock()
        self.collection_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.queue_lock = threading.Lock()  
        
        
        self.pages_processed = 0
        self.pages_added_to_db = 0
        self.errors_count = 0
        self.urls_filtered_out = 0  
        
        
        self.stop_crawling = threading.Event()
        
        
        self.enable_mistral_processing = enable_mistral_processing
        self.processor_agent = None
        if self.enable_mistral_processing:
            try:
                from agents.agent_processor import ProcessorAgent
                self.processor_agent = ProcessorAgent()
                print("✅ Procesamiento con Mistral habilitado")
            except Exception as e:
                print(f"⚠️ No se pudo habilitar el procesamiento con Mistral: {e}")
                self.enable_mistral_processing = False
        
        
        self.enable_gliner_processing = False
        self.gliner_agent = None
        
        
        self.mistral_processed = 0
        self.mistral_errors = 0
        self.gliner_processed = 0
        self.gliner_errors = 0
        
        
        self.chunks_file_path = self._initialize_chunks_file()
        self.file_lock = threading.Lock()
    
    def enable_gliner(self):
        """Habilita el procesamiento con GLiNER"""
        if not self.enable_gliner_processing:
            try:
                from agents.agent_gliner import GLiNERAgent
                self.gliner_agent = GLiNERAgent()
                self.enable_gliner_processing = True
                print("✅ Procesamiento con GLiNER habilitado")
                return True
            except Exception as e:
                print(f"⚠️ No se pudo habilitar el procesamiento con GLiNER: {e}")
                self.enable_gliner_processing = False
                return False
        return True
    
    def disable_mistral(self):
        """Deshabilita el procesamiento con Mistral"""
        self.enable_mistral_processing = False
        self.processor_agent = None
        print("❌ Procesamiento con Mistral deshabilitado")
    
    def _initialize_chunks_file(self) -> str:
        """Inicializa el archivo para guardar los chunks"""
        try:
            
            logs_dir = "crawler_logs"
            logs_path = os.path.abspath(logs_dir)
            
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
                print(f"📁 Directorio creado: {logs_path}")
            
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(logs_path, f"chunks_{timestamp}.txt")
            
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"CHUNKS DE CRAWLER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
            
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"📄 Archivo de chunks creado exitosamente:")
                print(f"   📍 Ruta: {filename}")
                print(f"   📏 Tamaño inicial: {file_size} bytes")
            else:
                print(f"⚠️ Advertencia: El archivo {filename} no se pudo verificar")
            
            return filename
            
        except Exception as e:
            print(f"❌ Error al inicializar archivo de chunks: {e}")
            import traceback
            traceback.print_exc()
            
            return "chunks_error.txt"
    
    def _save_chunk_to_file(self, doc_id: str, content: str, metadata: dict, processor: str):
        """Guarda un chunk en el archivo de texto EXACTAMENTE como se almacena en ChromaDB"""
        print(f"💾 Intentando guardar chunk en archivo: {self.chunks_file_path}")
        print(f"   - ID: {doc_id}")
        print(f"   - Procesador: {processor}")
        print(f"   - Tamaño contenido: {len(content)} caracteres")
        
        with self.file_lock:
            try:
                
                if not os.path.exists(self.chunks_file_path):
                    print(f"⚠️ El archivo no existe, creándolo: {self.chunks_file_path}")
                    os.makedirs(os.path.dirname(self.chunks_file_path), exist_ok=True)
                
                with open(self.chunks_file_path, 'a', encoding='utf-8') as f:
                    f.write("=" * 100 + "\n")
                    f.write("INICIO DE CHUNK - EXACTAMENTE COMO SE GUARDA EN CHROMADB\n")
                    f.write("=" * 100 + "\n\n")
                    
                    f.write("### INFORMACIÓN DEL DOCUMENTO ###\n")
                    f.write(f"ID: {doc_id}\n")
                    f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"PROCESADOR: {processor}\n\n")
                    
                    f.write("### METADATA (como se guarda en ChromaDB) ###\n")
                    f.write("{\n")
                    for key, value in sorted(metadata.items()):
                        if isinstance(value, str) and '\n' in value:
                            
                            f.write(f'  "{key}": {json.dumps(value, ensure_ascii=False)},\n')
                        else:
                            f.write(f'  "{key}": {json.dumps(value, ensure_ascii=False)},\n')
                    f.write("}\n\n")
                    
                    f.write("### CONTENIDO DEL DOCUMENTO (como se guarda en ChromaDB) ###\n")
                    f.write(f"Tamaño: {len(content)} caracteres\n")
                    f.write("-" * 80 + "\n")
                    f.write(content)
                    f.write("\n" + "-" * 80 + "\n\n")
                    
                    f.write("### RESUMEN DE INFORMACIÓN CLAVE ###\n")
                    
                    if processor == "GLiNER":
                        if 'entities_data' in metadata:
                            try:
                                entities_data = json.loads(metadata['entities_data'])
                                if 'entities' in entities_data:
                                    entities = entities_data['entities']
                                    if 'countries' in entities and entities['countries']:
                                        f.write(f"PAÍSES DETECTADOS: {', '.join(entities['countries'])}\n")
                                    if 'cities' in entities and entities['cities']:
                                        f.write(f"CIUDADES DETECTADAS: {', '.join(entities['cities'])}\n")
                                    if 'hotels' in entities and entities['hotels']:
                                        hotel_names = [h.get('name', 'Sin nombre') for h in entities['hotels']]
                                        f.write(f"HOTELES DETECTADOS: {', '.join(hotel_names[:10])}\n")
                            except:
                                pass
                    
                    elif processor == "Mistral":
                        if 'structured_data' in metadata:
                            try:
                                structured_data = json.loads(metadata['structured_data'])
                                if 'pais' in structured_data:
                                    f.write(f"PAÍS: {structured_data['pais']}\n")
                                if 'ciudad' in structured_data:
                                    f.write(f"CIUDAD: {structured_data['ciudad']}\n")
                                if 'lugares' in structured_data and structured_data['lugares']:
                                    lugares_nombres = [lugar.get('nombre', '') for lugar in structured_data['lugares'][:10] if lugar.get('nombre')]
                                    if lugares_nombres:
                                        f.write(f"LUGARES PRINCIPALES: {', '.join(lugares_nombres)}\n")
                                    tipos_unicos = list(set([lugar.get('tipo', '') for lugar in structured_data['lugares'] if lugar.get('tipo')]))
                                    if tipos_unicos:
                                        f.write(f"TIPOS DE LUGARES: {', '.join(tipos_unicos)}\n")
                            except:
                                pass
                    
                    elif processor == "ACO":
                        if 'keywords_used' in metadata:
                            f.write(f"PALABRAS CLAVE UTILIZADAS: {metadata['keywords_used']}\n")
                        if 'extraction_method' in metadata:
                            f.write(f"MÉTODO DE EXTRACCIÓN: {metadata['extraction_method']}\n")
                    
                    f.write("\n" + "=" * 100 + "\n")
                    f.write("FIN DE CHUNK\n")
                    f.write("=" * 100 + "\n\n\n")
                    
                print(f"✅ Chunk guardado exitosamente en archivo")
                
                
                if os.path.exists(self.chunks_file_path):
                    file_size = os.path.getsize(self.chunks_file_path)
                    print(f"   📏 Tamaño actual del archivo: {file_size:,} bytes")
                    
            except Exception as e:
                print(f"❌ Error guardando chunk en archivo: {e}")
                import traceback
                traceback.print_exc()

    def is_valid_url(self, url: str) -> bool:
        """Determina si una URL es válida para el crawler de turismo."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
        except:
            return False

        
        unwanted_extensions = [
            '.css', '.js', '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.webp', '.ico', '.xml', '.zip'
        ]
        if any(url.lower().endswith(ext) for ext in unwanted_extensions):
            return False

        
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
            return True  
        
        
        combined_text = f"{url.lower()} {text_content.lower()}"
        
        
        expanded_keywords = self._expand_keywords(self.current_query_keywords)
        
        
        for keyword in expanded_keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower and keyword_lower in combined_text:
                return True
        
        return False
    
    def _expand_keywords(self, keywords: List[str]) -> List[str]:
        """
        Expande las palabras clave con sinónimos y variaciones para mejorar el matching.
        """
        expanded = set(keywords)  
        
        
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
            
            
            if keyword_lower in synonyms:
                expanded.update(synonyms[keyword_lower])
            
            
            if keyword_lower.endswith('s') and len(keyword_lower) > 3:
                expanded.add(keyword_lower[:-1])  
            else:
                expanded.add(keyword_lower + 's')  
            
            
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
        
        
        title_text = a_tag.get('title', '')
        
        
        parent_text = ""
        if a_tag.parent:
            parent_text = a_tag.parent.get_text(strip=True)[:100]  
        
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
                
                link_text = self._extract_link_text(a_tag)
                if self._has_common_keywords(absolute_url, link_text):
                    links.append(absolute_url)
                else:
                    
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
        
        
        with self.visited_lock:
            if url in self.visited_urls:
                return None
            self.visited_urls.add(url)
        
        
        with self.stats_lock:
            self.pages_processed += 1
            current_processed = self.pages_processed
            
            
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
                
                if self.enable_gliner_processing and self.gliner_agent:
                    try:
                        
                        gliner_response = self.gliner_agent.receive({
                            'type': 'process_content',
                            'content_data': content_data
                        }, self)
                        
                        if gliner_response.get('success') and gliner_response.get('data'):
                            processed_data = gliner_response['data']
                            
                            
                            with self.collection_lock:
                                
                                structured_text = self._format_gliner_data(processed_data)
                                
                                doc_id = f"gliner_doc_{hash(url) % 10000000}_{depth}_{int(time.time())}"
                                
                                
                                metadata = {
                                    "url": content_data["url"],
                                    "title": content_data["title"],
                                    "source": "parallel_tourism_crawler",
                                    "depth": depth,
                                    "thread_id": str(thread_id),
                                    "processed_by_gliner": True,
                                    "entities_data": json.dumps(processed_data, ensure_ascii=False)
                                }
                                
                                
                                if 'entities' in processed_data:
                                    entities = processed_data['entities']
                                    if 'countries' in entities and entities['countries']:
                                        metadata['countries'] = ', '.join(entities['countries'])
                                    if 'cities' in entities and entities['cities']:
                                        metadata['cities'] = ', '.join(entities['cities'])
                                    if 'hotels' in entities and entities['hotels']:
                                        hotel_names = [h['name'] for h in entities['hotels']]
                                        metadata['hotels'] = ', '.join(hotel_names[:5])
                                
                                
                                print(f"\n📝 GUARDANDO CHUNK EN CHROMADB:")
                                print(f"   📌 ID: {doc_id}")
                                print(f"   🔗 URL: {content_data['url']}")
                                print(f"   📄 Título: {content_data['title']}...")
                                print(f"   📏 Tamaño del texto: {len(structured_text)} caracteres")
                                print(f"   🏷️ Procesado por: GLiNER")
                                if 'countries' in metadata:
                                    print(f"   🌍 Países: {metadata['countries']}")
                                if 'cities' in metadata:
                                    print(f"   🏙️ Ciudades: {metadata['cities']}")
                                print(f"   📊 Metadata: {len(metadata)} campos")
                                print(f"   ✅ Chunk guardado exitosamente\n")
                                
                                self.collection.add(
                                    documents=[structured_text],
                                    metadatas=[metadata],
                                    ids=[doc_id]
                                )
                                
                                
                                self._save_chunk_to_file(doc_id, structured_text, metadata, "GLiNER")
                            
                            with self.stats_lock:
                                self.pages_added_to_db += 1
                                self.gliner_processed += 1
                            
                            print(f"[Thread-{thread_id}] ✅ Contenido procesado con GLiNER: {content_data['title'][:50]}...")
                        else:
                            
                            self._save_original_content(content_data, depth, thread_id)
                            
                    except Exception as e:
                        print(f"[Thread-{thread_id}] ⚠️ Error en procesamiento GLiNER: {e}")
                        with self.stats_lock:
                            self.gliner_errors += 1
                        
                        self._save_original_content(content_data, depth, thread_id)
                
                
                elif self.enable_mistral_processing and self.processor_agent:
                    try:
                        
                        processor_response = self.processor_agent.receive({
                            'type': 'process_content',
                            'content_data': content_data
                        }, self)
                        
                        if processor_response.get('success') and processor_response.get('data'):
                            processed_data = processor_response['data']
                            
                            
                            with self.collection_lock:
                                
                                structured_text = self._format_structured_data(processed_data)
                                
                                doc_id = f"mistral_doc_{hash(url) % 10000000}_{depth}_{int(time.time())}"
                                
                                
                                metadata = {
                                    "url": content_data["url"],
                                    "title": content_data["title"],
                                    "source": "parallel_tourism_crawler",
                                    "depth": depth,
                                    "thread_id": str(thread_id),
                                    "processed_by_mistral": True,
                                    "structured_data": json.dumps(processed_data, ensure_ascii=False)
                                }
                                
                                
                                if 'pais' in processed_data:
                                    metadata['pais'] = processed_data['pais']
                                if 'ciudad' in processed_data:
                                    metadata['ciudad'] = processed_data['ciudad']
                                
                                
                                if 'lugares' in processed_data and processed_data['lugares']:
                                    tipos_lugares = list(set([lugar.get('tipo', '') for lugar in processed_data['lugares'] if lugar.get('tipo')]))
                                    if tipos_lugares:
                                        metadata['tipos_lugares'] = ', '.join(tipos_lugares)
                                    
                                    
                                    nombres_lugares = [lugar.get('nombre', '') for lugar in processed_data['lugares'][:5] if lugar.get('nombre')]
                                    if nombres_lugares:
                                        metadata['lugares_principales'] = ', '.join(nombres_lugares)
                                
                                
                                print(f"\n📝 GUARDANDO CHUNK EN CHROMADB:")
                                print(f"   📌 ID: {doc_id}")
                                print(f"   🔗 URL: {content_data['url']}")
                                print(f"   📄 Título: {content_data['title']}...")
                                print(f"   📏 Tamaño del texto: {len(structured_text)} caracteres")
                                print(f"   🏷️ Procesado por: Mistral")
                                if 'paises' in metadata:
                                    print(f"   🌍 Países: {metadata['paises']}")
                                print(f"   📊 Metadata: {len(metadata)} campos")
                                print(f"   ✅ Chunk guardado exitosamente\n")
                                
                                self.collection.add(
                                    documents=[structured_text],
                                    metadatas=[metadata],
                                    ids=[doc_id]
                                )
                                
                                
                                self._save_chunk_to_file(doc_id, structured_text, metadata, "GLiNER")
                            
                            with self.stats_lock:
                                self.pages_added_to_db += 1
                                self.mistral_processed += 1
                            
                            print(f"[Thread-{thread_id}] ✅ Contenido procesado con Mistral: {content_data['title'][:50]}...")
                        else:
                            
                            self._save_original_content(content_data, depth, thread_id)
                            
                    except Exception as e:
                        print(f"[Thread-{thread_id}] ⚠️ Error en procesamiento Mistral: {e}")
                        with self.stats_lock:
                            self.mistral_errors += 1
                        
                        self._save_original_content(content_data, depth, thread_id)
                else:
                    
                    self._save_original_content(content_data, depth, thread_id)
                
                
                new_links = []
                if depth < self.max_depth:
                    links = self.get_links(url, soup)
                    
                    filtered_links = links[:10]  
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
                
                
                completed_futures = []
                for future in active_futures:
                    if future.done():
                        completed_futures.append(future)
                        try:
                            result = future.result()
                            if result and result.get("success"):
                                
                                for new_link_data in result.get("new_links", []):
                                    new_url, new_depth = new_link_data
                                    with self.visited_lock:
                                        if new_url not in self.visited_urls:
                                            self.urls_to_visit.put((new_url, new_depth))
                        except Exception as e:
                            print(f"Error procesando resultado: {str(e)}")
                
                
                for future in completed_futures:
                    active_futures.remove(future)
                
                
                while (len(active_futures) < self.num_threads and 
                       not self.urls_to_visit.empty() and 
                       not self.stop_crawling.is_set()):
                    
                    try:
                        url_data = self.urls_to_visit.get_nowait()
                        future = executor.submit(self._process_single_url, url_data)
                        active_futures.append(future)
                    except queue.Empty:
                        break
                
                
                current_time = time.time()
                if current_time - last_progress_time > 5.0:
                    elapsed = current_time - start_time
                    rate = self.pages_processed / elapsed if elapsed > 0 else 0
                    print(f"📈 Progreso: {self.pages_processed}/{self.max_pages} páginas "
                          f"({self.pages_added_to_db} añadidas a DB, {self.errors_count} errores) "
                          f"- {rate:.1f} páginas/seg - {len(active_futures)} hilos activos")
                    last_progress_time = current_time
                
                
                if not active_futures and self.urls_to_visit.empty():
                    break
                
                
                time.sleep(0.1)
                
                
                if time.time() - start_time > 300:  
                    print("⏰ Timeout de seguridad alcanzado, finalizando crawler...")
                    self.stop_crawling.set()
                    break
        
        
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
        if self.enable_mistral_processing:
            print(f"   • Páginas procesadas con Mistral: {self.mistral_processed}")
            print(f"   • Errores de procesamiento Mistral: {self.mistral_errors}")
            if self.processor_agent:
                stats = self.processor_agent.get_stats()
                print(f"   • Tasa de éxito Mistral: {stats['success_rate']:.2%}")
        if self.enable_gliner_processing:
            print(f"   • Páginas procesadas con GLiNER: {self.gliner_processed}")
            print(f"   • Errores de procesamiento GLiNER: {self.gliner_errors}")
            if self.gliner_agent:
                stats = self.gliner_agent.get_stats()
                print(f"   • Tasa de éxito GLiNER: {stats['success_rate']:.2%}")
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

        
        if improved_query:
            print(f"🔍 Búsqueda con consulta mejorada: '{improved_query}'")
            
            search_keywords = [improved_query]
        else:
            print(f"🔍 Búsqueda para palabras clave del usuario: {keywords}")
            search_keywords = keywords
        
        
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
        
        
        print("\n❌ No se pudieron obtener resultados de ningún motor de búsqueda")
        print(f"   - Palabras clave utilizadas: {search_keywords}")
        print("   - Todos los motores de búsqueda fallaron o están bloqueados")
        return []
    
    def _direct_keyword_search(self, keywords: list) -> list:
        """
        Búsqueda directa construyendo URLs basadas en palabras clave.
        """
        direct_urls = []
        
        
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
        
        
        for keyword in keywords[:3]:  
            keyword_encoded = keyword.replace(' ', '+')
            
            for template in url_templates:
                try:
                    url = template.format(keyword=keyword_encoded)
                    direct_urls.append(url)
                except:
                    continue
        
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            
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
        
        return list(set(direct_urls))  
    
    def _search_duckduckgo_links(self, keywords: list, num_results_per_query: int = 8) -> list:
        """
        Realiza búsquedas usando la librería duckduckgo_search que maneja mejor los CAPTCHAs.
        """
        all_urls = set()
        
        try:
            
            from duckduckgo_search import DDGS
            
            print("  🦆 Usando DuckDuckGo Search API...")
            
            
            with DDGS() as ddgs:
                
                query = ' '.join(keywords)
                print(f"  🔍 Buscando: '{query}'")
                
                try:
                    
                    results = list(ddgs.text(
                        query, 
                        max_results=num_results_per_query,
                        safesearch='off',
                        region='wt-wt'  
                    ))
                    
                    
                    for result in results:
                        if 'href' in result:
                            all_urls.add(result['href'])
                    
                    print(f"    ✓ Encontrados {len(all_urls)} resultados")
                    
                except Exception as e:
                    print(f"    ❌ Error en búsqueda: {e}")
            
            return list(all_urls)
            
        except ImportError:
            print("  ⚠️ Librería duckduckgo_search no disponible, usando método alternativo...")
            
            return self._search_with_requests(keywords, num_results_per_query)
    
    def _search_with_requests(self, keywords: list, num_results_per_query: int = 8) -> list:
        """
        Método de búsqueda alternativo usando requests directamente.
        """
        import random
        from urllib.parse import quote_plus
        
        all_urls = set()
        
        
        search_queries = [' '.join(keywords)]  
        
        if len(keywords) > 1:
            search_queries.extend(keywords[:2])  
        
        
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
        
        for query in search_queries[:5]:  
            try:
                print(f"  🔍 Buscando: '{query}'")
                
                
                encoded_query = quote_plus(query)
                search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
                
                
                response = search_session.get(search_url, timeout=30, allow_redirects=True)
                
                
                if response.status_code in [200, 202, 301, 302]:
                    
                    if response.status_code in [301, 302] and 'location' in response.headers:
                        redirect_url = response.headers['location']
                        response = search_session.get(redirect_url, timeout=30)
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    
                    if len(response.text) < 1000:
                        print(f"    ⚠️ Respuesta muy corta: {len(response.text)} caracteres")
                    
                    
                    results_found = 0
                    
                    
                    result_divs = soup.find_all('div', class_='result')
                    print(f"    📦 Encontrados {len(result_divs)} divs de resultados")
                    
                    for div in result_divs:
                        
                        link = div.find('a', class_='result__a')
                        if link and link.get('href'):
                            href = link['href']
                            
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
                    
                    
                    if results_found == 0:
                        all_links = soup.find_all('a', href=True)
                        print(f"    🔗 Total de enlaces en la página: {len(all_links)}")
                    
                    for link in all_links:
                        href = link.get('href', '')
                        
                        
                        if 'duckduckgo.com/l/?uddg=' in href:
                            try:
                                from urllib.parse import parse_qs, urlparse, unquote
                                parsed = urlparse(href)
                                params = parse_qs(parsed.query)
                                if 'uddg' in params:
                                    href = unquote(params['uddg'][0])
                            except:
                                continue
                        
                        
                        if (href and 
                            href.startswith('http') and 
                            'duckduckgo.com' not in href and
                            len(href) > 15):
                            
                            
                            all_urls.add(href)
                            results_found += 1
                            
                            if results_found >= num_results_per_query:
                                break
                    
                    print(f"    ✓ Encontrados {results_found} resultados")
                else:
                    print(f"    ⚠️ Respuesta HTTP {response.status_code}")
                
                
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
                
                
                
                for li in soup.find_all('li', class_='b_algo'):
                    h2 = li.find('h2')
                    if h2:
                        link = h2.find('a', href=True)
                        if link and link['href'].startswith('http'):
                            all_urls.add(link['href'])
                
                
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
                
                
                if len(all_urls) < 5:
                    
                    url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
                    potential_urls = url_pattern.findall(str(soup))
                    for url in potential_urls:
                        if ('bing.com' not in url and 
                            'microsoft.com' not in url and
                            len(url) > 30 and
                            url.count('/') >= 3):  
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
                        break  
                        
            except Exception as e:
                continue  
        
        if not all_urls:
            print("    ⚠️ No se encontraron resultados en ninguna instancia de Searx")
        
        return list(all_urls)
    
    def _fallback_direct_search(self, keywords: list, num_results_per_query: int = 10) -> list:
        """
        Método de respaldo que construye URLs directamente basándose en las palabras clave.
        """
        direct_urls = []
        
        
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
        
        
        for keyword in keywords:
            keyword_encoded = keyword.replace(' ', '+')
            for site_name, url_template in tourism_sites.items():
                try:
                    url = url_template.format(keyword_encoded)
                    direct_urls.append(url)
                except:
                    continue
        
        
        query = ' '.join(keywords).lower()
        
        
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
        
        
        unique_urls = list(dict.fromkeys(direct_urls))  
        
        print(f"    ✓ Generadas {len(unique_urls)} URLs directas")
        return unique_urls[:num_results_per_query]
    
    def _alternative_search(self, keywords: list) -> set:
        """
        Método alternativo de búsqueda usando Google Search (via SerpAPI simulado o scraping básico)
        """
        from urllib.parse import quote_plus
        alternative_urls = set()
        
        try:
            
            for keyword in keywords[:3]:  
                try:
                    query = f"{keyword} tourism travel guide"
                    encoded_query = quote_plus(query)
                    
                    
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
                        
                        import re
                        
                        url_pattern = r'href="(https?://[^"]+)"'
                        found_urls = re.findall(url_pattern, response.text)
                        
                        for url in found_urls:
                            
                            if ('google.com' not in url and 
                                'googleusercontent' not in url and
                                self._is_tourism_relevant_url(url)):
                                alternative_urls.add(url)
                                if len(alternative_urls) >= 15:
                                    break
                    
                    time.sleep(2)  
                    
                except Exception as e:
                    print(f"    Error en búsqueda alternativa para '{keyword}': {e}")
                    continue
                    
        except Exception as e:
            print(f"  Error general en búsqueda alternativa: {e}")
        
        
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
        
        
        tourism_terms = ['tourism', 'travel', 'visit', 'guide', 'attractions', 'hotels']
        
        for keyword in keywords:
            
            queries.append(f"{keyword} tourism")
            queries.append(f"{keyword} travel guide")
            queries.append(f"visit {keyword}")
            
            
            keyword_lower = keyword.lower()
            if any(term in keyword_lower for term in ['hotel', 'accommodation', 'hospedaje']):
                queries.append(f"best hotels {keyword}")
                queries.append(f"{keyword} booking")
            elif any(term in keyword_lower for term in ['restaurant', 'food', 'comida']):
                queries.append(f"best restaurants {keyword}")
            else:
                queries.append(f"{keyword} attractions")
        
        
        if len(keywords) > 1:
            combined = ' '.join(keywords[:2])  
            queries.append(f"{combined} tourism")
        
        return queries
    
    def _is_tourism_relevant_url(self, url: str) -> bool:
        """
        Verifica si una URL es relevante para turismo (para búsquedas web).
        Versión mejorada con criterios más flexibles.
        """
        url_lower = url.lower()
        
        
        tourism_domains = [
            'tripadvisor', 'booking', 'expedia', 'hotels', 'airbnb',
            'lonelyplanet', 'frommers', 'roughguides', 'nationalgeographic',
            'timeout', 'viator', 'getyourguide', 'agoda', 'hostelworld',
            'kayak', 'trivago', 'travelocity', 'orbitz', 'priceline',
            'marriott', 'hilton', 'hyatt', 'ihg', 'accor'
        ]
        
        
        tourism_patterns = [
            'tourism', 'travel', 'vacation', 'destination', 'attractions',
            'things-to-do', 'guide', 'visit', 'hotel', 'restaurant',
            'turismo', 'viaje', 'vacaciones', 'destino', 'atracciones',
            'hoteles', 'restaurante', 'hospedaje', 'alojamiento',
            'resort', 'lodge', 'inn', 'hostel', 'accommodation',
            'sightseeing', 'tour', 'excursion', 'adventure', 'explore',
            'beach', 'playa', 'museum', 'museo', 'park', 'parque'
        ]
        
        
        if any(domain in url_lower for domain in tourism_domains):
            return True
        
        
        if any(pattern in url_lower for pattern in tourism_patterns):
            return True
        
        
        if self.current_query_keywords:
            for keyword in self.current_query_keywords:
                if keyword.lower() in url_lower:
                    return True
        
        
        
        unwanted_extensions = ['.css', '.js', '.jpg', '.jpeg', '.png', '.gif', '.pdf']
        unwanted_patterns = ['/login', '/signin', '/register', '/api/', '/cdn-cgi/']
        
        if not any(ext in url_lower for ext in unwanted_extensions) and \
           not any(pattern in url_lower for pattern in unwanted_patterns):
            
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
            'angoola': [  
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
        
        
        self.current_query_keywords = keywords
        print(f"🎯 Filtrado de URLs activado para palabras clave: {keywords}")
        
        
        self.urls_filtered_out = 0
        
        
        initial_urls = self.google_search_links(keywords, num_results=20, improved_query=improved_query)
        
        if not initial_urls:
            print("❌ No se encontraron URLs iniciales")
            return 0
        
        print(f"✅ Encontradas {len(initial_urls)} URLs iniciales")
        
        
        while not self.urls_to_visit.empty():
            try:
                self.urls_to_visit.get_nowait()
            except queue.Empty:
                break
        
        
        for url in initial_urls:
            self.urls_to_visit.put((url, 0))
        
        
        original_max_depth = self.max_depth
        self.max_depth = max_depth
        
        
        result = self.run_parallel_crawler()
        
        
        self.max_depth = original_max_depth
        
        return result

    
    def crawl_from_links(self, links: list, max_depth: int = 2):
        """Método de compatibilidad - redirige al crawler paralelo"""
        print("⚠️ Usando crawler paralelo en lugar del método legacy crawl_from_links")
        
        
        while not self.urls_to_visit.empty():
            try:
                self.urls_to_visit.get_nowait()
            except queue.Empty:
                break
        
        
        for url in links[:20]:  
            self.urls_to_visit.put((url, 0))
        
        
        return self.run_parallel_crawler()

    def run_crawler(self):
        """Método de compatibilidad - redirige al crawler paralelo"""
        print("⚠️ Usando crawler paralelo en lugar del método secuencial legacy")
        return self.run_parallel_crawler()
    
    def _save_original_content(self, content_data: Dict, depth: int, thread_id: int):
        """Guarda el contenido original sin procesar con Mistral"""
        with self.collection_lock:
            doc_id = f"parallel_doc_{hash(content_data['url']) % 10000000}_{depth}_{int(time.time())}"
            
            
            metadata = {
                "url": content_data["url"],
                "title": content_data["title"],
                "source": "parallel_tourism_crawler",
                "depth": depth,
                "thread_id": str(thread_id),
                "processed_by_mistral": False
            }
            
            
            print(f"\n📝 GUARDANDO CHUNK EN CHROMADB:")
            print(f"   📌 ID: {doc_id}")
            print(f"   🔗 URL: {content_data['url']}")
            print(f"   📄 Título: {content_data['title']}...")
            print(f"   📏 Tamaño del texto: {len(content_data['content'])} caracteres")
            print(f"   🏷️ Procesado por: Crawler (sin Mistral)")
            print(f"   📊 Profundidad: {depth}")
            print(f"   🧵 Thread ID: {thread_id}")
            print(f"   ✅ Chunk guardado exitosamente\n")
            
            self.collection.add(
                documents=[content_data["content"]],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            
            self._save_chunk_to_file(doc_id, content_data["content"], metadata, "Crawler (sin procesamiento)")
        
        with self.stats_lock:
            self.pages_added_to_db += 1
        
        print(f"[Thread-{thread_id}] ✓ Contenido añadido (sin Mistral): {content_data['title'][:50]}...")
    
    def _format_structured_data(self, processed_data: Dict) -> str:
        """
        Convierte los datos estructurados en un texto formateado para embeddings
        """
        formatted_text = []
        
        
        if 'source_title' in processed_data:
            formatted_text.append(f"Título: {processed_data['source_title']}")
        
        
        if 'pais' in processed_data:
            formatted_text.append(f"\nPaís: {processed_data['pais']}")
        if 'ciudad' in processed_data:
            formatted_text.append(f"Ciudad: {processed_data['ciudad']}")
        
        
        if 'lugares' in processed_data and processed_data['lugares']:
            formatted_text.append("\nLugares:")
            for lugar in processed_data['lugares']:
                lugar_info = []
                
                
                if 'nombre' in lugar:
                    lugar_info.append(f"\n- {lugar['nombre']}")
                if 'tipo' in lugar:
                    lugar_info.append(f"  Tipo: {lugar['tipo']}")
                if 'subtipo' in lugar:
                    lugar_info.append(f" - {lugar['subtipo']}")
                
                
                if 'ubicacion' in lugar:
                    ubicacion = lugar['ubicacion']
                    if 'zona' in ubicacion:
                        lugar_info.append(f"  Zona: {ubicacion['zona']}")
                    if 'direccion' in ubicacion:
                        lugar_info.append(f"  Dirección: {ubicacion['direccion']}")
                
                
                if 'descripcion' in lugar:
                    lugar_info.append(f"  Descripción: {lugar['descripcion']}")
                
                
                if 'precios' in lugar:
                    precios = lugar['precios']
                    precio_info = []
                    if 'rango_precio' in precios:
                        precio_info.append(f"Rango: {precios['rango_precio']}")
                    if 'precio_promedio' in precios:
                        precio_info.append(f"Promedio: {precios['precio_promedio']}")
                    if 'precio_desde' in precios:
                        precio_info.append(f"Desde: {precios['precio_desde']}")
                    if precio_info:
                        lugar_info.append(f"  Precios: {', '.join(precio_info)}")
                
                
                if 'calificacion' in lugar:
                    calif = lugar['calificacion']
                    if 'puntuacion' in calif:
                        lugar_info.append(f"  Calificación: {calif['puntuacion']}/{calif.get('escala', '5')}")
                
                
                if 'servicios' in lugar and lugar['servicios']:
                    lugar_info.append(f"  Servicios: {', '.join(lugar['servicios'])}")
                
                formatted_text.extend(lugar_info)
        
        
        if 'informacion_general' in processed_data:
            info = processed_data['informacion_general']
            formatted_text.append("\nInformación General:")
            
            if 'clima' in info:
                formatted_text.append(f"- Clima: {info['clima']}")
            if 'mejor_epoca_visita' in info:
                formatted_text.append(f"- Mejor época para visitar: {info['mejor_epoca_visita']}")
            if 'presupuesto_diario' in info:
                formatted_text.append(f"- Presupuesto diario: {info['presupuesto_diario']}")
            if 'duracion_recomendada' in info:
                formatted_text.append(f"- Duración recomendada: {info['duracion_recomendada']}")
            if 'tips_viajeros' in info and info['tips_viajeros']:
                formatted_text.append(f"- Tips: {', '.join(info['tips_viajeros'])}")
        
        
        if 'actividades_populares' in processed_data and processed_data['actividades_populares']:
            formatted_text.append("\nActividades Populares:")
            for actividad in processed_data['actividades_populares']:
                if 'nombre' in actividad:
                    act_info = [f"- {actividad['nombre']}"]
                    if 'precio' in actividad:
                        act_info.append(f"(Precio: {actividad['precio']})")
                    if 'duracion' in actividad:
                        act_info.append(f"(Duración: {actividad['duracion']})")
                    formatted_text.append(' '.join(act_info))
        
        
        if 'gastronomia' in processed_data:
            gastro = processed_data['gastronomia']
            if 'platos_tipicos' in gastro and gastro['platos_tipicos']:
                formatted_text.append(f"\nPlatos típicos: {', '.join(gastro['platos_tipicos'])}")
            if 'bebidas_tipicas' in gastro and gastro['bebidas_tipicas']:
                formatted_text.append(f"Bebidas típicas: {', '.join(gastro['bebidas_tipicas'])}")
        
        
        if 'tipo_contenido' in processed_data and processed_data['tipo_contenido'] == 'no_turistico':
            return "Contenido no relacionado con turismo"
        
        
        if not formatted_text and isinstance(processed_data, dict):
            for key, value in processed_data.items():
                if key not in ['source_url', 'source_title', 'processed_by'] and value:
                    formatted_text.append(f"{key}: {value}")
        
        return '\n'.join(formatted_text) if formatted_text else "Sin información estructurada disponible"
    
    def _format_gliner_data(self, processed_data: Dict) -> str:
        """
        Convierte los datos extraídos por GLiNER en un texto formateado para embeddings
        """
        formatted_text = []
        
        
        if 'source_title' in processed_data:
            formatted_text.append(f"Título: {processed_data['source_title']}")
        
        
        if 'summary' in processed_data:
            formatted_text.append(f"\nResumen: {processed_data['summary']}")
        
        
        if 'entities' in processed_data:
            entities = processed_data['entities']
            
            
            if 'countries' in entities and entities['countries']:
                formatted_text.append(f"\nPaíses: {', '.join(entities['countries'])}")
            
            
            if 'cities' in entities and entities['cities']:
                formatted_text.append(f"Ciudades: {', '.join(entities['cities'])}")
            
            
            if 'destinations' in entities and entities['destinations']:
                dest_text = []
                for dest in entities['destinations']:
                    dest_info = f"{dest['name']}"
                    if 'type' in dest:
                        dest_info += f" ({dest['type']})"
                    dest_text.append(dest_info)
                formatted_text.append(f"\nDestinos turísticos: {', '.join(dest_text)}")
            
            
            if 'hotels' in entities and entities['hotels']:
                hotel_text = []
                for hotel in entities['hotels']:
                    hotel_info = f"{hotel['name']}"
                    if 'type' in hotel:
                        hotel_info += f" ({hotel['type']})"
                    hotel_text.append(hotel_info)
                formatted_text.append(f"\nHoteles y alojamientos: {', '.join(hotel_text)}")
            
            
            if 'attractions' in entities and entities['attractions']:
                attr_text = []
                for attr in entities['attractions']:
                    attr_info = f"{attr['name']}"
                    if 'type' in attr:
                        attr_info += f" ({attr['type']})"
                    attr_text.append(attr_info)
                formatted_text.append(f"\nAtracciones turísticas: {', '.join(attr_text)}")
            
            
            if 'restaurants' in entities and entities['restaurants']:
                rest_text = []
                for rest in entities['restaurants']:
                    rest_info = f"{rest['name']}"
                    if 'type' in rest:
                        rest_info += f" ({rest['type']})"
                    rest_text.append(rest_info)
                formatted_text.append(f"\nRestaurantes: {', '.join(rest_text)}")
            
            
            if 'prices' in entities and entities['prices']:
                price_text = []
                for price in entities['prices']:
                    price_text.append(price['text'])
                formatted_text.append(f"\nInformación de precios: {', '.join(price_text)}")
            
            
            if 'activities' in entities and entities['activities']:
                act_text = []
                for act in entities['activities']:
                    act_info = f"{act['name']}"
                    if 'type' in act:
                        act_info += f" ({act['type']})"
                    act_text.append(act_info)
                formatted_text.append(f"\nActividades: {', '.join(act_text)}")
            
            
            if 'transport' in entities and entities['transport']:
                trans_text = []
                for trans in entities['transport']:
                    trans_info = f"{trans['name']}"
                    if 'type' in trans:
                        trans_info += f" ({trans['type']})"
                    trans_text.append(trans_info)
                formatted_text.append(f"\nTransporte: {', '.join(trans_text)}")
            
            
            if 'organizations' in entities and entities['organizations']:
                org_text = []
                for org in entities['organizations']:
                    org_info = f"{org['name']}"
                    if 'type' in org:
                        org_info += f" ({org['type']})"
                    org_text.append(org_info)
                formatted_text.append(f"\nOrganizaciones turísticas: {', '.join(org_text)}")
        
        
        if not formatted_text and 'raw_entities' in processed_data:
            raw_entities = processed_data['raw_entities']
            if raw_entities:
                entity_texts = []
                for entity in raw_entities[:20]:  
                    entity_texts.append(f"{entity['text']} ({entity['type']})")
                formatted_text.append(f"Entidades encontradas: {', '.join(entity_texts)}")
        
        return '\n'.join(formatted_text) if formatted_text else "Sin entidades relevantes encontradas"