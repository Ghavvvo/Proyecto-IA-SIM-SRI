from autogen import Agent
from typing import List
import google.generativeai as genai

class CoordinatorAgent(Agent):
    def __init__(self, name, crawler_agent, rag_agent, interface_agent, context_agent, route_agent):
        super().__init__(name)
        self.crawler_agent = crawler_agent
        self.rag_agent = rag_agent
        self.interface_agent = interface_agent
        self.context_agent = context_agent
        self.route_agent = route_agent

    def start(self): 
        self._notify_interface('system_start', {
            'component': self.name,
            'action': 'initializing'
        })
        # Verificar si la colección ya tiene datos antes de correr el crawler
        if hasattr(self.crawler_agent.crawler.collection, 'count'):
            try:
                count = self.crawler_agent.crawler.collection.count()
            except Exception:
                count = 0
        else:
            # Fallback si no existe el método count
            try:
                results = self.crawler_agent.crawler.collection.query(query_texts=["test"], n_results=1)
                count = len(results.get('documents', [[]])[0])
            except Exception:
                count = 0
        if count > 0:
            # Si ya hay datos, inicializar el RAG directamente
            self._notify_interface('rag_init', {
                'action': 'initializing_with_existing_data'
            })
            self.rag_agent.receive({'type': 'init_collection', 'collection': self.crawler_agent.crawler.collection},
                                   self)
        else:            # Si no hay datos, correr el crawler
            crawl_result = self.crawler_agent.receive({'type': 'crawl'}, self)
            self._notify_interface('crawler_start', {
                'reason': 'no_existing_data'
            })
            if crawl_result['type'] == 'crawled':
                self.rag_agent.receive({'type': 'init_collection', 'collection': crawl_result['collection']}, self)
                

    def ask(self, query):
        # Paso 0: Manejar solicitudes directas de rutas
        direct_route_response = self._handle_direct_route_request(query)
        if direct_route_response:
            return direct_route_response
        
        # Paso 1: Analizar y mejorar la consulta usando el contexto
        print("🧠 Analizando consulta con contexto conversacional...")
        context_analysis = self.context_agent.receive({'type': 'analyze_query', 'query': query}, self)
        
        if context_analysis['type'] == 'query_analyzed':
            analysis = context_analysis['analysis']
            improved_query = analysis['improved_query']
            context_info = analysis['context_analysis']
            
            print(f"📝 Consulta original: {query}")
            print(f"🔍 Consulta mejorada: {improved_query}")
            print(f"🎯 Intención detectada: {context_info.get('user_intent', 'No detectada')}")
            print(f"🔗 Continuación de tema: {'Sí' if context_info.get('is_continuation', False) else 'No'}")
            
            if analysis.get('improvements_made'):
                print(f"✨ Mejoras aplicadas: {', '.join(analysis['improvements_made'])}")
        else:
            # Si hay error en el análisis, usar la consulta original
            improved_query = query
            print("⚠️ Error en análisis de contexto, usando consulta original")

        # Paso 2: Consultar al agente RAG con la consulta mejorada
        self._notify_interface('query_received', {
            'query': improved_query,
            'original_query': query,
            'status': 'processing'
        })
        response = self.rag_agent.receive({'type': 'query', 'query': improved_query}, self)

        if response['type'] == 'answer':
            # Paso 3: Guardar la interacción en el contexto
            final_answer = response['answer']
            self.context_agent.receive({
                'type': 'add_interaction', 
                'query': query, 
                'response': final_answer
            }, self)
            
            # Utilizar Gemini para evaluar si la respuesta es útil
            evaluation = self._evaluate_response_usefulness(query, final_answer)
            if not evaluation:
                # Extraer palabras clave problemáticas cuando la consulta no es relevante
                problematic_keywords = self._extract_problematic_keywords(query, response['answer'])

                print(f"La respuesta proporcionada por el sistema RAG no es útil para la consulta.")
                print(f"Palabras clave problemáticas identificadas: {', '.join(problematic_keywords)}")
                print("🐜 Iniciando búsqueda inteligente con Ant Colony Optimization...")

                # NUEVO FLUJO: Búsqueda en Google + Exploración ACO
                # Paso 1: Buscar en Google y explorar con ACO
                aco_result = self.crawler_agent.receive({
                    'type': 'search_google_aco', 
                    'keywords': problematic_keywords,
                    'max_urls': 15,
                    'max_depth': 2
                }, self)
                
                if aco_result.get('type') == 'aco_completed' and aco_result.get('content_extracted'):
                    content_count = aco_result.get('content_extracted', 0)
                    aco_stats = aco_result.get('aco_statistics', {})
                    
                    print(f"🐜 Exploración ACO completada exitosamente")
                    print(f"📊 Contenido extraído: {content_count} páginas")
                    print(f"🎯 Tasa de éxito ACO: {aco_stats.get('success_rate', 0)*100:.1f}%")
                    print(f"🕸️ Senderos de feromonas creados: {aco_stats.get('pheromone_trails_count', 0)}")
                    
                    if content_count > 0:
                        # Paso 2: Consultar nuevamente al RAG con la información actualizada
                        print("🔄 Consultando RAG con información obtenida por ACO...")
                        new_response = self.rag_agent.receive({'type': 'query', 'query': query}, self)
                        
                        if new_response['type'] == 'answer':
                            print("✅ Nueva respuesta generada con información de exploración ACO")
                            final_answer = new_response['answer']
                        else:
                            return new_response.get('msg', 'Error en la nueva consulta después de exploración ACO')
                    else:
                        print("⚠️ ACO no extrajo contenido útil, intentando método alternativo...")
                        # Fallback a método anterior
                        return self._fallback_search_method(problematic_keywords, query)
                else:
                    print("❌ Error en exploración ACO, intentando método alternativo...")
                    return self._fallback_search_method(problematic_keywords, query)

            # Manejar sugerencia de ruta si es relevante
            final_answer = self._suggest_route_if_relevant(query, final_answer)
            
            return final_answer

        return response.get('msg', 'Error al consultar la base de datos')
    
    def get_conversation_stats(self):
        """
        Obtiene estadísticas de la conversación actual.
        
        Returns:
            Diccionario con estadísticas de conversación
        """
        return self.context_agent.get_conversation_stats()
    
    def clear_conversation_context(self):
        """
        Limpia el contexto de conversación.
        
        Returns:
            Resultado de la operación de limpieza
        """
        result = self.context_agent.receive({'type': 'clear_context'}, self)
        if result['type'] == 'context_cleared':
            print("🧹 Contexto de conversación limpiado exitosamente")
            return True
        return False
    
    def get_conversation_context(self):
        """
        Obtiene el contexto actual de conversación.
        
        Returns:
            Contexto de conversación actual
        """
        result = self.context_agent.receive({'type': 'get_context'}, self)
        return result if result['type'] == 'context_data' else None

    def _evaluate_response_usefulness(self, query, answer):
        """
        Evalúa si la respuesta del RAG es útil para la consulta del usuario.

        Args:
            query (str): La consulta original del usuario.
            answer (str): La respuesta generada por el sistema RAG.

        Returns:
            bool: True si la respuesta es útil, False en caso contrario.
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Eres un evaluador de respuestas. Determina si la siguiente respuesta es útil para la consulta del usuario.

            Consulta del usuario: {query}

            Respuesta generada: {answer}

            INSTRUCCIONES:
            - Evalúa solamente si la respuesta proporciona información relevante para la consulta
            - Una respuesta útil contiene información específica relacionada con la consulta
            - Una respuesta no útil puede ser vaga, no relacionada o simplemente indicar que no hay suficiente información
            - Responde únicamente con 'true' si la respuesta es útil, o 'false' si no lo es.
            """

            response = model.generate_content(prompt)
            result = response.text.lower().strip()

            # Determinar si la respuesta es útil basada en el texto generado
            return 'true' in result
        except Exception as e:
            print(f"Error al evaluar la utilidad de la respuesta: {e}")
            # En caso de error, asumimos que la respuesta es útil
            return True

    def _extract_problematic_keywords(self, query, answer):
        """
        Extrae las palabras clave que hicieron que la consulta no fuera relevante.

        Args:
            query (str): La consulta original del usuario.
            answer (str): La respuesta generada por el sistema RAG.

        Returns:
            List[str]: Lista de palabras clave problemáticas.
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Eres un analizador de consultas. Tu tarea es identificar las palabras clave específicas en la consulta del usuario que causaron que el sistema no pudiera proporcionar una respuesta útil.

            Consulta del usuario: {query}

            Respuesta del sistema: {answer}

            INSTRUCCIONES:
            - Identifica las palabras clave, términos específicos, nombres de lugares, actividades o conceptos en la consulta que el sistema no pudo manejar adecuadamente
            - Enfócate en sustantivos, nombres propios, actividades específicas, y términos técnicos que parecen estar fuera del alcance de la base de datos
            - No incluyas palabras comunes como artículos, preposiciones o verbos generales
            - Devuelve ÚNICAMENTE las palabras clave separadas por comas, sin explicaciones adicionales
            - Si no hay palabras problemáticas específicas, responde con "ninguna"

            Palabras clave problemáticas:"""

            response = model.generate_content(prompt)
            result = response.text.strip()

            # Procesar la respuesta
            if result.lower() == "ninguna" or not result:
                return []
            
            # Dividir por comas y limpiar espacios
            keywords = [keyword.strip() for keyword in result.split(',') if keyword.strip()]
            
            return keywords

        except Exception as e:
            print(f"Error al extraer palabras clave problemáticas: {e}")
            return []
    
    def _fallback_search_method(self, problematic_keywords, query):
        """
        Método de fallback cuando ACO falla.
        """
        print("🔄 Ejecutando método de búsqueda alternativo...")
        
        # Usar el método anterior como fallback
        crawl_result = self.crawler_agent.receive({'type': 'crawl_keywords', 'keywords': problematic_keywords}, self)
        
        if crawl_result.get('type') == 'crawled':
            pages_processed = crawl_result.get('pages_processed', 0)
            print(f"�� Base de datos actualizada con {pages_processed} páginas usando método alternativo")
            
            new_response = self.rag_agent.receive({'type': 'query', 'query': query}, self)
            if new_response['type'] == 'answer':
                return new_response['answer']
            else:
                return new_response.get('msg', 'Error en la nueva consulta después de actualizar la base de datos')
        else:
            error_msg = crawl_result.get('msg', 'Error desconocido')
            print(f"❌ No se pudo actualizar la base de datos: {error_msg}")
            return "Lo siento, no pude encontrar información relevante para tu consulta en este momento."
        
    def _notify_interface(self, event_type, event_data):
        """Envía notificaciones sin system_state"""

        message = {
            'event_type': event_type,
            'event_data': event_data or {}
        }
        return self.interface_agent.receive(message, self)

    def _suggest_route_if_relevant(self, query: str, current_answer: str) -> str:
        """Sugiere una ruta optimizada si es relevante para la consulta"""
        # Paso 1: Detectar si debemos ofrecer ruta
        offer_decision = self.context_agent.receive({
            'type': 'should_offer_route',
            'query': query,
            'response': current_answer
        }, self)
        
        should_offer = False
        if offer_decision['type'] == 'route_offer_decision':
            should_offer = offer_decision['should_offer']
        
        if not should_offer:
            return current_answer
        
        # Paso 2: Extraer lugares relevantes
        extraction_result = self.context_agent.receive({
            'type': 'extract_relevant_places',
            'response': current_answer
        }, self)
        
        places = []
        if extraction_result['type'] == 'extracted_places':
            places = extraction_result['places']
        
        # Necesitamos al menos 2 lugares para generar una ruta
        if len(places) < 2:
            return current_answer
        
        # Paso 3: Confirmar con el usuario
        print(current_answer)
        print("\n¿Desea generar una ruta optimizada para visitar estos lugares? (sí/no)")
        user_response = input("> ").strip().lower()
        
        if user_response not in ['sí', 'si', 's', 'yes', 'y']:
            return current_answer
        
        # Paso 4: Generar y mostrar ruta
        route_result = self.route_agent.receive({
            'type': 'optimize_route',
            'places': places
        }, self)
        
        if route_result['type'] != 'route_result':
            return current_answer
        
        # Paso 5: Formatear y añadir la ruta a la respuesta
        route_str = self._format_route(route_result)
        return current_answer + f"\n\n{route_str}"

    def _format_route(self, route_result):
        """Formatea los resultados de la ruta para mostrar al usuario"""
        return (
            "🗺️ Ruta optimizada:\n" +
            "\n".join([f"{i+1}. {place}" for i, place in enumerate(route_result['order'])]) +
            f"\n\n📏 Distancia total: {route_result['total_distance_km']} km" +
            f"\n⏱️ Tiempo estimado: {route_result['total_distance_km']/5:.1f} horas (caminando)"
        )
    
    def _handle_direct_route_request(self, query):
        """Maneja solicitudes directas de rutas turísticas"""
        if any(keyword in query.lower() for keyword in ['ruta para visitar', 'recorrido para visitar']):
            extraction_result = self.context_agent.receive({
                'type': 'extract_relevant_places',
                'response': query
            }, self)
            
            places = []
            if extraction_result['type'] == 'extracted_places':
                places = extraction_result['places']
            
            if len(places) >= 2:
                route_result = self.route_agent.receive({
                    'type': 'optimize_route',
                    'places': places
                }, self)
                
                if route_result['type'] == 'route_result':
                    return self._format_route(route_result)
            else:
                return "Por favor, proporcione al menos dos lugares para generar una ruta."
        return None