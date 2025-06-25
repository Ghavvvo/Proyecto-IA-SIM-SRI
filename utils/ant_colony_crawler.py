"""
Implementación del Algoritmo de Colonia de Hormigas (ACO) para Web Crawling
Optimiza la exploración de URLs basándose en feromonas y heurísticas
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class URLNode:
    """Representa un nodo (URL) en el grafo de exploración"""
    url: str
    pheromone: float = 1.0
    heuristic_value: float = 0.0
    visited_count: int = 0
    content_quality: float = 0.0
    depth: int = 0
    parent_url: Optional[str] = None
    keywords_found: List[str] = None
    
    def __post_init__(self):
        if self.keywords_found is None:
            self.keywords_found = []


class AntColonyOptimizer:
    """Implementación del algoritmo de colonia de hormigas para web crawling"""
    
    def __init__(self, 
                 num_ants: int = 10,
                 alpha: float = 1.0,      
                 beta: float = 2.0,       
                 rho: float = 0.1,        
                 q: float = 100.0,        
                 max_iterations: int = 5,
                 max_depth: int = 3):
        
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        
        
        self.nodes: Dict[str, URLNode] = {}
        self.adjacency_list: Dict[str, List[str]] = {}
        self.best_paths: List[List[str]] = []
        self.iteration_stats: List[Dict] = []
        
        
        self.lock = threading.Lock()
        
        print(f"🐜 ACO Inicializado:")
        print(f"   • Hormigas: {num_ants}")
        print(f"   • Alpha (feromonas): {alpha}")
        print(f"   • Beta (heurística): {beta}")
        print(f"   • Rho (evaporación): {rho}")
        print(f"   • Iteraciones máximas: {max_iterations}")
    
    def calculate_url_heuristic(self, url: str, keywords: List[str]) -> float:
        """
        Calcula el valor heurístico de una URL basado en palabras clave y patrones
        """
        url_lower = url.lower()
        heuristic = 0.0
        
        
        keyword_score = 0.0
        for keyword in keywords:
            if keyword.lower() in url_lower:
                keyword_score += 1.0
        
        
        if keywords:
            keyword_score = keyword_score / len(keywords)
        
        
        valuable_patterns = {
            '/destination': 0.8,
            '/travel': 0.7,
            '/tourism': 0.9,
            '/tour': 0.6,
            '/visit': 0.7,
            '/hotel': 0.8,
            '/restaurant': 0.6,
            '/attraction': 0.8,
            '/guide': 0.7,
            '/review': 0.5,
            'tripadvisor': 0.9,
            'booking': 0.8,
            'lonelyplanet': 0.8,
            'expedia': 0.7
        }
        
        pattern_score = 0.0
        for pattern, weight in valuable_patterns.items():
            if pattern in url_lower:
                pattern_score += weight
        
        
        unwanted_patterns = [
            '/login', '/signup', '/cart', '/admin', '/api/',
            '.css', '.js', '.jpg', '.png', '.pdf'
        ]
        
        penalty = 0.0
        for pattern in unwanted_patterns:
            if pattern in url_lower:
                penalty += 0.5
        
        
        path_depth = len([p for p in urlparse(url).path.split('/') if p])
        depth_penalty = min(path_depth * 0.1, 0.5)
        
        
        heuristic = (keyword_score * 0.4 + 
                    pattern_score * 0.4 + 
                    (1.0 - depth_penalty) * 0.2 - 
                    penalty)
        
        return max(0.1, min(1.0, heuristic))  
    
    def extract_links_from_url(self, url: str, keywords: List[str]) -> List[str]:
        """
        Extrae enlaces de una URL con filtrado inteligente
        """
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href')
                if not href or href.startswith('#'):
                    continue
                
                absolute_url = urljoin(url, href)
                
                
                if not self._is_valid_url(absolute_url):
                    continue
                
                
                link_text = a_tag.get_text(strip=True)
                title = a_tag.get('title', '')
                combined_text = f"{absolute_url} {link_text} {title}".lower()
                
                
                has_keywords = any(keyword.lower() in combined_text for keyword in keywords)
                
                if has_keywords or self._has_tourism_patterns(absolute_url):
                    links.append(absolute_url)
            
            return list(set(links))[:20]  
            
        except Exception as e:
            print(f"Error extrayendo enlaces de {url}: {e}")
            return []
    
    def _is_valid_url(self, url: str) -> bool:
        """Verifica si una URL es válida para crawling"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
        except:
            return False
        
        
        unwanted_extensions = [
            '.css', '.js', '.jpg', '.jpeg', '.png', '.gif', 
            '.pdf', '.svg', '.mp3', '.mp4', '.zip'
        ]
        
        return not any(url.lower().endswith(ext) for ext in unwanted_extensions)
    
    def _has_tourism_patterns(self, url: str) -> bool:
        """Verifica si la URL tiene patrones relacionados con turismo"""
        tourism_patterns = [
            'tourism', 'travel', 'hotel', 'restaurant', 'attraction',
            'destination', 'vacation', 'trip', 'guide', 'visit'
        ]
        
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in tourism_patterns)
    
    def calculate_transition_probability(self, current_url: str, next_url: str) -> float:
        """
        Calcula la probabilidad de transición de una URL a otra
        """
        current_node = self.nodes.get(current_url)
        next_node = self.nodes.get(next_url)
        
        if not current_node or not next_node:
            return 0.0
        
        
        pheromone_component = next_node.pheromone ** self.alpha
        
        
        heuristic_component = next_node.heuristic_value ** self.beta
        
        
        probability = pheromone_component * heuristic_component
        
        return probability
    
    def select_next_url(self, current_url: str, available_urls: List[str]) -> Optional[str]:
        """
        Selecciona la siguiente URL usando probabilidades ACO
        """
        if not available_urls:
            return None
        
        
        probabilities = []
        for url in available_urls:
            prob = self.calculate_transition_probability(current_url, url)
            probabilities.append(prob)
        
        
        total_prob = sum(probabilities)
        if total_prob == 0:
            
            return random.choice(available_urls)
        
        normalized_probs = [p / total_prob for p in probabilities]
        
        
        r = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(normalized_probs):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return available_urls[i]
        
        
        return available_urls[-1]
    
    def ant_exploration(self, start_url: str, keywords: List[str], ant_id: int) -> List[str]:
        """
        Simula el recorrido de una hormiga
        """
        path = [start_url]
        current_url = start_url
        visited_in_path = {start_url}
        
        for step in range(self.max_depth):
            
            if current_url not in self.adjacency_list:
                links = self.extract_links_from_url(current_url, keywords)
                with self.lock:
                    self.adjacency_list[current_url] = links
                    
                    
                    for link in links:
                        if link not in self.nodes:
                            heuristic = self.calculate_url_heuristic(link, keywords)
                            self.nodes[link] = URLNode(
                                url=link,
                                heuristic_value=heuristic,
                                depth=step + 1,
                                parent_url=current_url
                            )
            
            
            available_urls = [
                url for url in self.adjacency_list.get(current_url, [])
                if url not in visited_in_path
            ]
            
            if not available_urls:
                break
            
            
            next_url = self.select_next_url(current_url, available_urls)
            
            if next_url:
                path.append(next_url)
                visited_in_path.add(next_url)
                current_url = next_url
                
                
                with self.lock:
                    if next_url in self.nodes:
                        self.nodes[next_url].visited_count += 1
            else:
                break
        
        return path
    
    def evaluate_path_quality(self, path: List[str], keywords: List[str]) -> float:
        """
        Evalúa la calidad de un camino basado en contenido extraído
        """
        total_quality = 0.0
        content_extracted = 0
        
        for url in path:
            node = self.nodes.get(url)
            if node:
                
                heuristic_score = node.heuristic_value
                keyword_score = len(node.keywords_found) / max(len(keywords), 1)
                depth_penalty = 1.0 - (node.depth * 0.1)
                
                url_quality = (heuristic_score * 0.5 + 
                              keyword_score * 0.3 + 
                              depth_penalty * 0.2)
                
                total_quality += url_quality
                content_extracted += 1
        
        
        if len(path) > 0:
            return total_quality / len(path)
        
        return 0.0
    
    def update_pheromones(self, paths: List[List[str]], qualities: List[float]):
        """
        Actualiza las feromonas basándose en la calidad de los caminos
        """
        
        with self.lock:
            for node in self.nodes.values():
                node.pheromone *= (1.0 - self.rho)
                node.pheromone = max(0.1, node.pheromone)  
        
        
        for path, quality in zip(paths, qualities):
            pheromone_deposit = self.q * quality
            
            for url in path:
                with self.lock:
                    if url in self.nodes:
                        self.nodes[url].pheromone += pheromone_deposit
                        
                        self.nodes[url].pheromone = min(10.0, self.nodes[url].pheromone)
    
    def run_optimization(self, start_urls: List[str], keywords: List[str]) -> Dict:
        """
        Ejecuta el algoritmo de optimización de colonia de hormigas
        """
        print(f"🐜 Iniciando optimización ACO con {len(start_urls)} URLs iniciales")
        
        
        for url in start_urls:
            heuristic = self.calculate_url_heuristic(url, keywords)
            self.nodes[url] = URLNode(
                url=url,
                heuristic_value=heuristic,
                depth=0
            )
        
        best_overall_quality = 0.0
        best_overall_paths = []
        
        for iteration in range(self.max_iterations):
            print(f"🔄 Iteración ACO {iteration + 1}/{self.max_iterations}")
            
            iteration_paths = []
            iteration_qualities = []
            
            
            with ThreadPoolExecutor(max_workers=min(self.num_ants, 5)) as executor:
                futures = []
                
                for ant_id in range(self.num_ants):
                    start_url = random.choice(start_urls)
                    future = executor.submit(self.ant_exploration, start_url, keywords, ant_id)
                    futures.append(future)
                
                
                for future in as_completed(futures):
                    try:
                        path = future.result()
                        if len(path) > 1:  
                            quality = self.evaluate_path_quality(path, keywords)
                            iteration_paths.append(path)
                            iteration_qualities.append(quality)
                    except Exception as e:
                        print(f"Error en exploración de hormiga: {e}")
            
            
            if iteration_paths:
                self.update_pheromones(iteration_paths, iteration_qualities)
                
                
                best_iteration_quality = max(iteration_qualities)
                best_iteration_idx = iteration_qualities.index(best_iteration_quality)
                best_iteration_path = iteration_paths[best_iteration_idx]
                
                if best_iteration_quality > best_overall_quality:
                    best_overall_quality = best_iteration_quality
                    best_overall_paths = [best_iteration_path]
                
                
                avg_quality = np.mean(iteration_qualities)
                self.iteration_stats.append({
                    'iteration': iteration + 1,
                    'paths_found': len(iteration_paths),
                    'avg_quality': avg_quality,
                    'best_quality': best_iteration_quality,
                    'nodes_discovered': len(self.nodes)
                })
                
                print(f"   • Caminos encontrados: {len(iteration_paths)}")
                print(f"   • Calidad promedio: {avg_quality:.3f}")
                print(f"   • Mejor calidad: {best_iteration_quality:.3f}")
                print(f"   • Nodos descubiertos: {len(self.nodes)}")
            
            else:
                print(f"   • No se encontraron caminos válidos")
        
        
        total_nodes = len(self.nodes)
        total_edges = sum(len(edges) for edges in self.adjacency_list.values())
        avg_pheromone = np.mean([node.pheromone for node in self.nodes.values()]) if self.nodes else 0
        
        results = {
            'best_paths': best_overall_paths,
            'best_quality': best_overall_quality,
            'total_nodes_discovered': total_nodes,
            'total_edges_discovered': total_edges,
            'average_pheromone_level': avg_pheromone,
            'iteration_stats': self.iteration_stats,
            'pheromone_trails_count': total_edges,
            'success_rate': best_overall_quality,
            'nodes_discovered': total_nodes,
            'average_path_length': np.mean([len(path) for path in best_overall_paths]) if best_overall_paths else 0
        }
        
        print(f"🎯 Optimización ACO completada:")
        print(f"   • Mejor calidad: {best_overall_quality:.3f}")
        print(f"   • Nodos descubiertos: {total_nodes}")
        print(f"   • Aristas descubiertas: {total_edges}")
        print(f"   • Nivel promedio de feromonas: {avg_pheromone:.3f}")
        
        return results


def extract_content_from_url(url: str, keywords: List[str]) -> Optional[Dict]:
    """
    Extrae contenido de una URL específica
    """
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        
        title = soup.title.string if soup.title else ""
        title = title.strip()
        
        
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        
        content_candidates = []
        
        
        for tag in ['article', 'main', 'section']:
            elements = soup.find_all(tag)
            content_candidates.extend(elements)
        
        
        if not content_candidates:
            content_candidates = soup.find_all('p')
        
        
        content_text = ""
        for candidate in content_candidates:
            text = candidate.get_text(separator=' ', strip=True)
            if text and len(text) > 50:
                content_text += text + " "
        
        
        content_text = ' '.join(content_text.split())
        
        if len(content_text) < 100:
            return None
        
        
        keywords_found = []
        content_lower = content_text.lower()
        for keyword in keywords:
            if keyword.lower() in content_lower:
                keywords_found.append(keyword)
        
        return {
            'url': url,
            'title': title,
            'content': content_text[:2000],  
            'keywords_found': keywords_found,
            'extraction_method': 'aco'
        }
        
    except Exception as e:
        print(f"Error extrayendo contenido de {url}: {e}")
        return None


def integrate_aco_with_crawler(crawler, keywords: List[str], max_urls: int = 15, improved_query: str = None, max_depth: int = 2) -> List[Dict]:
    """
    Integra ACO con el crawler existente para búsqueda optimizada
    
    Args:
        crawler: Instancia del crawler
        keywords: Lista de palabras clave para la búsqueda
        max_urls: Número máximo de URLs a procesar
        improved_query: Consulta mejorada por el agente de contexto (opcional)
        max_depth: Profundidad máxima de exploración (se incrementa en cada iteración)
    """
    print(f"🐜 Integrando ACO con crawler para palabras clave: {keywords}")
    if improved_query:
        print(f"🔍 Con consulta mejorada: '{improved_query}'")
    print(f"🔢 Profundidad de exploración: {max_depth}")
    
    
    aco = AntColonyOptimizer(
        num_ants=8,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        max_iterations=3,
        max_depth=max_depth  
    )
    
    
    initial_urls = crawler.google_search_links(keywords, num_results=10, improved_query=improved_query)
    
    if not initial_urls:
        print("❌ No se encontraron URLs iniciales para ACO")
        return []
    
    print(f"🔍 URLs iniciales para ACO: {len(initial_urls)}")
    
    
    aco_results = aco.run_optimization(initial_urls, keywords)
    
    
    extracted_content = []
    urls_to_extract = set()
    
    
    for path in aco_results['best_paths']:
        urls_to_extract.update(path)
    
    
    if not urls_to_extract:
        sorted_nodes = sorted(
            aco.nodes.items(),
            key=lambda x: x[1].pheromone * x[1].heuristic_value,
            reverse=True
        )
        urls_to_extract = {url for url, _ in sorted_nodes[:max_urls]}
    
    
    urls_to_extract = list(urls_to_extract)[:max_urls]
    
    print(f"📄 Extrayendo contenido de {len(urls_to_extract)} URLs optimizadas por ACO")
    
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(extract_content_from_url, url, keywords): url 
            for url in urls_to_extract
        }
        
        for future in as_completed(futures):
            try:
                content = future.result()
                if content:
                    extracted_content.append(content)
            except Exception as e:
                print(f"Error extrayendo contenido: {e}")
    
    print(f"✅ ACO extrajo contenido de {len(extracted_content)} páginas")
    
    return extracted_content


def test_aco_performance():
    """
    Función de prueba para evaluar el rendimiento de ACO
    """
    print("🧪 PRUEBA DE RENDIMIENTO ACO")
    print("=" * 50)
    
    
    aco = AntColonyOptimizer(
        num_ants=5,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        max_iterations=2,
        max_depth=2
    )
    
    
    test_urls = [
        "https://www.tripadvisor.com/",
        "https://www.booking.com/",
        "https://www.lonelyplanet.com/"
    ]
    
    test_keywords = ["hotel", "tourism", "travel"]
    
    
    start_time = time.time()
    results = aco.run_optimization(test_urls, test_keywords)
    end_time = time.time()
    
    print(f"\n📊 Resultados de la prueba:")
    print(f"   • Tiempo de ejecución: {end_time - start_time:.2f} segundos")
    print(f"   • Nodos descubiertos: {results['total_nodes_discovered']}")
    print(f"   • Aristas descubiertas: {results['total_edges_discovered']}")
    print(f"   • Mejor calidad: {results['best_quality']:.3f}")
    print(f"   • Caminos encontrados: {len(results['best_paths'])}")
    
    return results


if __name__ == "__main__":
    test_aco_performance()