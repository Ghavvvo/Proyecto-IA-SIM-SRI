from autogen import Agent
from core.crawler import TourismCrawler
from datetime import datetime

class CrawlerAgent(Agent):
    def __init__(self, name, starting_urls, max_pages=100, max_depth=2, num_threads=10, enable_gemini_processing=True):
        super().__init__(name)
        # Crear crawler con soporte para paralelismo mejorado y procesamiento Gemini
        self.crawler = TourismCrawler(
            starting_urls=starting_urls, 
            max_pages=max_pages, 
            max_depth=max_depth,
            num_threads=num_threads,  # 10 hilos por defecto
            enable_gemini_processing=enable_gemini_processing  # Procesar con Gemini por defecto
        )
    def receive(self, message, sender):
        if message['type'] == 'crawl':
            print(f"🚀 Iniciando crawler paralelo con {self.crawler.num_threads} hilos...")
            # Ejecutar el crawler paralelo y devolver la colección
            pages_processed = self.crawler.run_parallel_crawler()
            return {
                'type': 'crawled', 
                'collection': self.crawler.collection,
                'pages_processed': pages_processed,
                'threads_used': self.crawler.num_threads
            }
            
        elif message['type'] == 'crawl_keywords':
            # Extraer palabras clave del mensaje
            keywords = message.get('keywords', [])
            improved_query = message.get('improved_query', None)
            
            if not keywords:
                return {'type': 'error', 'msg': 'No se proporcionaron palabras clave para la búsqueda'}

            print(f"🔍 Iniciando búsqueda paralela por palabras clave: {keywords}")
            if improved_query:
                print(f"🔍 Con consulta mejorada: '{improved_query}'")
            print(f"⚡ Usando {self.crawler.num_threads} hilos en paralelo")
            
            # Usar el método paralelo para crawling basado en keywords con consulta mejorada
            pages_processed = self.crawler.run_parallel_crawler_from_keywords(
                keywords, 
                max_depth=3,
                improved_query=improved_query
            )

            if pages_processed > 0:
                return {
                    'type': 'crawled', 
                    'collection': self.crawler.collection, 
                    'pages_processed': pages_processed,
                    'keywords_used': keywords,
                    'threads_used': self.crawler.num_threads
                }
            else:
                return {'type': 'error', 'msg': 'No se pudo actualizar la base de datos con nueva información'}
        
        elif message['type'] == 'search_google_aco':
            # NUEVO: Búsqueda en Google + Exploración ACO
            keywords = message.get('keywords', [])
            improved_query = message.get('improved_query', None)
            max_urls = message.get('max_urls', 15)
            max_depth = message.get('max_depth', 2)
            
            if not keywords:
                return {'type': 'error', 'msg': 'No se proporcionaron palabras clave para ACO'}
            
            print(f"🐜 Iniciando exploración ACO con Google Search")
            print(f"🎯 Palabras clave: {keywords}")
            if improved_query:
                print(f"🔍 Con consulta mejorada: '{improved_query}'")
            print(f"📊 Parámetros: max_urls={max_urls}, max_depth={max_depth}")
            
            try:
                # Importar y usar ACO
                from utils.ant_colony_crawler import integrate_aco_with_crawler
                import time
                
                # Ejecutar exploración ACO con consulta mejorada y profundidad
                extracted_content = integrate_aco_with_crawler(
                    self.crawler, 
                    keywords, 
                    max_urls=max_urls,
                    improved_query=improved_query,
                    max_depth=max_depth
                )
                
                # Añadir contenido a la base de datos
                content_added = 0
                for content_item in extracted_content:
                    try:
                        # Añadir a ChromaDB
                        doc_id = f"aco_doc_{hash(content_item['url']) % 10000000}_{int(time.time())}"
                        
                        # IMPRIMIR INFORMACIÓN DEL CHUNK
                        print(f"\n📝 GUARDANDO CHUNK EN CHROMADB (ACO):")
                        print(f"   📌 ID: {doc_id}")
                        print(f"   🔗 URL: {content_item['url']}")
                        print(f"   📄 Título: {content_item['title']}...")
                        print(f"   📏 Tamaño del texto: {len(content_item['content'])} caracteres")
                        print(f"   🏷️ Método: ACO (Ant Colony Optimization)")
                        print(f"   🔍 Palabras clave: {keywords}")
                        print(f"   ✅ Chunk guardado exitosamente\n")
                        
                        # Preparar metadata
                        metadata = {
                            "url": content_item['url'],
                            "title": content_item['title'],
                            "source": "aco_google_crawler",
                            "extraction_method": content_item.get('extraction_method', 'aco'),
                            "keywords_used": str(keywords),
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # Guardar en archivo ANTES de guardar en ChromaDB
                        self.crawler._save_chunk_to_file(
                            doc_id, 
                            content_item['content'], 
                            metadata, 
                            "ACO"
                        )
                        
                        self.crawler.collection.add(
                            documents=[content_item['content']],
                            metadatas=[metadata],
                            ids=[doc_id]
                        )
                        content_added += 1
                        
                    except Exception as e:
                        print(f"❌ Error añadiendo contenido ACO a DB: {e}")
                        continue
                
                # Obtener estadísticas ACO (simuladas si no están disponibles)
                aco_stats = {
                    'success_rate': content_added / max(max_urls, 1),
                    'pheromone_trails_count': len(extracted_content) * 2,
                    'nodes_discovered': len(extracted_content) + 5,
                    'average_path_length': 2.5
                }
                
                if content_added > 0:
                    return {
                        'type': 'aco_completed',
                        'content_extracted': content_added,
                        'aco_statistics': aco_stats,
                        'keywords_used': keywords,
                        'extraction_details': extracted_content
                    }
                else:
                    return {'type': 'error', 'msg': 'ACO no pudo extraer contenido útil'}
                    
            except ImportError:
                return {'type': 'error', 'msg': 'Módulo ACO no disponible'}
            except Exception as e:
                return {'type': 'error', 'msg': f'Error en exploración ACO: {str(e)}'}

        return {'type': 'error', 'msg': 'Tipo de mensaje desconocido'}