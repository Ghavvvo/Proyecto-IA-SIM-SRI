"""
Versión simplificada del TourismCrawler para experimentos de throughput.
Elimina dependencias de LLMs y simplifica el almacenamiento.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Optional, Set
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import queue
from datetime import datetime
import json


class SimplifiedCrawler:
    """Crawler simplificado para experimentos de rendimiento"""
    
    def __init__(self, starting_urls: List[str], max_pages: int = 100, 
                 max_depth: int = 3, num_threads: int = 10):
        self.starting_urls = starting_urls
        self.visited_urls = set()
        self.urls_to_visit = queue.Queue()
        
        # Configuración
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.num_threads = num_threads
        
        # Inicializar cola con URLs iniciales
        for url in starting_urls:
            self.urls_to_visit.put((url, 0))
        
        # Locks para thread safety
        self.visited_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Estadísticas
        self.pages_processed = 0
        self.pages_extracted = 0
        self.errors_count = 0
        self.urls_discovered = 0
        
        # Control de parada
        self.stop_crawling = threading.Event()
        
        # Métricas de rendimiento
        self.processing_times = []
        self.extraction_times = []
        self.request_times = []
        
        # Almacenamiento en memoria (sin ChromaDB)
        self.extracted_content = []
        self.content_lock = threading.Lock()
    
    def is_valid_url(self, url: str) -> bool:
        """Versión simplificada de validación de URL"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
        except:
            return False
        
        # Filtrar extensiones no deseadas
        unwanted_extensions = [
            '.css', '.js', '.jpg', '.jpeg', '.png', '.gif', '.pdf', 
            '.svg', '.mp3', '.mp4', '.avi', '.mov', '.webp', '.ico'
        ]
        if any(url.lower().endswith(ext) for ext in unwanted_extensions):
            return False
        
        # Filtrar patrones no deseados
        unwanted_patterns = [
            '/login', '/signin', '/signup', '/register', '/account',
            'javascript:', 'mailto:', 'tel:', '#'
        ]
        if any(pattern in url.lower() for pattern in unwanted_patterns):
            return False
        
        return True
    
    def clean_text(self, text: str) -> str:
        """Limpia el texto extraído"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,;:áéíóúÁÉÍÓÚñÑ-]', '', text)
        return text
    
    def extract_content(self, url: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Extrae contenido básico de la página"""
        extraction_start = time.time()
        
        try:
            # Extraer título
            title = soup.title.string if soup.title else ""
            title = self.clean_text(title)
            
            # Eliminar scripts y estilos
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Extraer texto principal
            content_text = ""
            
            # Buscar contenido en elementos principales
            for tag in ['article', 'main', 'section', 'div']:
                elements = soup.find_all(tag)
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if text and len(text) > 100:
                        content_text += text + " "
            
            # Si no hay suficiente contenido, usar párrafos
            if len(content_text) < 200:
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text:
                        content_text += text + " "
            
            content_text = self.clean_text(content_text)
            
            # Validar contenido mínimo
            if not title or not content_text or len(content_text) < 100:
                return None
            
            # Limitar tamaño del contenido
            if len(content_text) > 5000:
                content_text = content_text[:5000]
            
            extraction_time = time.time() - extraction_start
            with self.stats_lock:
                self.extraction_times.append(extraction_time)
            
            return {
                "url": url,
                "title": title,
                "content": content_text,
                "extraction_time": extraction_time
            }
            
        except Exception as e:
            return None
    
    def get_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extrae enlaces de la página"""
        links = []
        base_url = url
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if not href or href.startswith('#'):
                continue
            
            absolute_url = urljoin(base_url, href)
            if self.is_valid_url(absolute_url):
                links.append(absolute_url)
        
        # Limitar número de enlaces por página
        return list(set(links))[:20]
    
    def _process_single_url(self, url_data: tuple) -> Optional[Dict]:
        """Procesa una sola URL"""
        if self.stop_crawling.is_set():
            return None
        
        url, depth = url_data
        thread_id = threading.current_thread().ident
        process_start = time.time()
        
        # Verificar si ya fue visitada
        with self.visited_lock:
            if url in self.visited_urls:
                return None
            self.visited_urls.add(url)
        
        # Actualizar contador
        with self.stats_lock:
            self.pages_processed += 1
            current_processed = self.pages_processed
            
            if current_processed >= self.max_pages:
                self.stop_crawling.set()
                return None
        
        try:
            # Realizar petición HTTP
            request_start = time.time()
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; SimplifiedCrawler/1.0)'
            })
            request_time = time.time() - request_start
            
            with self.stats_lock:
                self.request_times.append(request_time)
            
            if response.status_code != 200:
                with self.stats_lock:
                    self.errors_count += 1
                return None
            
            # Parsear HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraer contenido
            content_data = self.extract_content(url, soup)
            
            if content_data:
                # Guardar contenido en memoria
                with self.content_lock:
                    self.extracted_content.append({
                        "url": url,
                        "title": content_data["title"],
                        "content_length": len(content_data["content"]),
                        "depth": depth,
                        "thread_id": str(thread_id),
                        "timestamp": datetime.now().isoformat()
                    })
                
                with self.stats_lock:
                    self.pages_extracted += 1
                
                # Extraer nuevos enlaces si no hemos alcanzado la profundidad máxima
                new_links = []
                if depth < self.max_depth:
                    links = self.get_links(url, soup)
                    new_links = [(link, depth + 1) for link in links]
                    
                    with self.stats_lock:
                        self.urls_discovered += len(links)
                
                # Registrar tiempo de procesamiento
                process_time = time.time() - process_start
                with self.stats_lock:
                    self.processing_times.append(process_time)
                
                return {
                    "url": url,
                    "title": content_data["title"],
                    "new_links": new_links,
                    "success": True,
                    "process_time": process_time
                }
            
            return None
            
        except Exception as e:
            with self.stats_lock:
                self.errors_count += 1
            return None
    
    def run_crawler(self) -> Dict:
        """Ejecuta el crawler y retorna métricas"""
        print(f"🚀 Iniciando crawler simplificado")
        print(f"   • Threads: {self.num_threads}")
        print(f"   • Max páginas: {self.max_pages}")
        print(f"   • Max profundidad: {self.max_depth}")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            active_futures = []
            
            while self.pages_processed < self.max_pages and not self.stop_crawling.is_set():
                # Procesar futuros completados
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
                            pass
                
                # Eliminar futuros completados
                for future in completed_futures:
                    active_futures.remove(future)
                
                # Añadir nuevas tareas
                while (len(active_futures) < self.num_threads and 
                       not self.urls_to_visit.empty() and 
                       not self.stop_crawling.is_set()):
                    try:
                        url_data = self.urls_to_visit.get_nowait()
                        future = executor.submit(self._process_single_url, url_data)
                        active_futures.append(future)
                    except queue.Empty:
                        break
                
                # Verificar si hay trabajo pendiente
                if not active_futures and self.urls_to_visit.empty():
                    break
                
                # Pequeña pausa para evitar busy waiting
                time.sleep(0.01)
        
        # Calcular métricas finales
        elapsed_time = time.time() - start_time
        
        return self._calculate_metrics(elapsed_time)
    
    def _calculate_metrics(self, elapsed_time: float) -> Dict:
        """Calcula métricas de rendimiento"""
        metrics = {
            "configuration": {
                "num_threads": self.num_threads,
                "max_pages": self.max_pages,
                "max_depth": self.max_depth,
                "starting_urls": len(self.starting_urls)
            },
            "results": {
                "pages_processed": self.pages_processed,
                "pages_extracted": self.pages_extracted,
                "errors_count": self.errors_count,
                "urls_discovered": self.urls_discovered,
                "urls_visited": len(self.visited_urls)
            },
            "performance": {
                "total_time": elapsed_time,
                "pages_per_second": self.pages_processed / elapsed_time if elapsed_time > 0 else 0,
                "average_process_time": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                "average_request_time": sum(self.request_times) / len(self.request_times) if self.request_times else 0,
                "average_extraction_time": sum(self.extraction_times) / len(self.extraction_times) if self.extraction_times else 0
            },
            "efficiency": {
                "extraction_rate": self.pages_extracted / self.pages_processed if self.pages_processed > 0 else 0,
                "error_rate": self.errors_count / self.pages_processed if self.pages_processed > 0 else 0,
                "thread_efficiency": (self.pages_processed / elapsed_time) / self.num_threads if elapsed_time > 0 else 0
            }
        }
        
        # Calcular percentiles si hay suficientes datos
        if len(self.processing_times) > 10:
            sorted_times = sorted(self.processing_times)
            metrics["performance"]["p50_process_time"] = sorted_times[len(sorted_times)//2]
            metrics["performance"]["p95_process_time"] = sorted_times[int(len(sorted_times)*0.95)]
            metrics["performance"]["p99_process_time"] = sorted_times[int(len(sorted_times)*0.99)]
        
        return metrics