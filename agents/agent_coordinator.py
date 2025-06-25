import json
import statistics
from autogen import Agent
from typing import List
from core.mistral_config import MistralClient
import math

from utils import (
    format_as_simulation_input,
    run_simulation_replicas,
    aggregate_simulation_results,
    format_aggregated_simulation_results,
    format_simulation_results
)


class CoordinatorAgent(Agent):
    def __init__(self, name, crawler_agent, rag_agent, interface_agent, context_agent, route_agent, tourist_guide_agent=None, simulation_agent=None):
        super().__init__(name)
        self.crawler_agent = crawler_agent
        self.rag_agent = rag_agent
        self.interface_agent = interface_agent
        self.context_agent = context_agent
        self.route_agent = route_agent
        self.tourist_guide_agent = tourist_guide_agent
        self.simulation_agent = simulation_agent
        
        
        self.planning_state = {
            'mode': 'normal',  
            'preferences': None,
            'aco_depth': 1,  
            'iterations': 0,
            'max_iterations': 5
        }

    def start(self):
        self._notify_interface('system_start', {
            'component': self.name,
            'action': 'initializing'
        })
        
        if hasattr(self.crawler_agent.crawler.collection, 'count'):
            try:
                count = self.crawler_agent.crawler.collection.count()
            except Exception:
                count = 0
        else:
            
            try:
                results = self.crawler_agent.crawler.collection.query(query_texts=["test"], n_results=1)
                count = len(results.get('documents', [[]])[0])
            except Exception:
                count = 0
        if count > 0:
            
            self.rag_agent.receive({'type': 'init_collection', 'collection': self.crawler_agent.crawler.collection},
                                   self)
        else:            
            crawl_result = self.crawler_agent.receive({'type': 'crawl'}, self)
            self._notify_interface('crawler_start', {
                'reason': 'no_existing_data'
            })
            if crawl_result['type'] == 'crawled':
                self.rag_agent.receive({'type': 'init_collection', 'collection': crawl_result['collection']}, self)


    def ask(self, query):
        
        if self.planning_state['mode'] == 'planning':
            return self._handle_planning_mode(query)

        
        user_intent = self._detect_user_intent(query)

        
        if user_intent == 'plan_vacation':
            return self._start_vacation_planning()

        elif user_intent == 'create_itinerary':
            return self._create_itinerary_with_current_info()

        elif user_intent == 'need_more_info':
            return self._search_more_information(query)

        
        if self._is_direct_route_request(query):
            return self._handle_direct_route_request(query)

        
        if self._is_route_confirmation(query):
            return self._generate_route_from_context()

        
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
            
            improved_query = query
            print("⚠️ Error en análisis de contexto, usando consulta original")

        
        self._notify_interface('query_received', {
            'query': improved_query,
            'original_query': query,
            'status': 'processing'
        })
        response = self.rag_agent.receive({'type': 'query', 'query': improved_query}, self)

        if response['type'] == 'answer':
            
            final_answer = response['answer']
            self.context_agent.receive({
                'type': 'add_interaction',
                'query': query,
                'response': final_answer
            }, self)

            
            evaluation = self._evaluate_response_usefulness(query, final_answer)
            if not evaluation:
                
                problematic_keywords = self._extract_problematic_keywords(query, response['answer'])

                print(f"La respuesta proporcionada por el sistema RAG no es útil para la consulta.")
                print(f"Palabras clave problemáticas identificadas: {', '.join(problematic_keywords)}")
                print("🐜 Iniciando búsqueda inteligente con Ant Colony Optimization...")

                
                
                if len(problematic_keywords) > 1:
                    
                    destination = None
                    interests = []

                    
                    for keyword in problematic_keywords:
                        
                        if any(place in keyword.lower() for place in ['cuba', 'habana', 'varadero', 'panama', 'angola', 'méxico', 'argentina', 'españa']):
                            destination = keyword
                        else:
                            interests.append(keyword)

                    
                    if not destination and len(problematic_keywords) > 1:
                        destination = problematic_keywords[0]
                        interests = problematic_keywords[1:]
                    elif not interests and destination:
                        interests = problematic_keywords

                    
                    search_queries = self._create_specific_search_queries(destination, interests)

                    print(f"📋 Se realizarán {len(search_queries)} búsquedas específicas:")
                    for i, query in enumerate(search_queries, 1):
                        print(f"   {i}. {query}")

                    total_content_extracted = 0

                    
                    for search_query in search_queries:
                        print(f"\n🔍 Buscando: '{search_query}'")

                        aco_result = self.crawler_agent.receive({
                            'type': 'search_google_aco',
                            'keywords': [search_query],
                            'improved_query': search_query,
                            'max_urls': 5,  
                            'max_depth': 2
                        }, self)

                        if aco_result.get('type') == 'aco_completed' and aco_result.get('content_extracted'):
                            content_count = aco_result.get('content_extracted', 0)
                            total_content_extracted += content_count
                            print(f"   ✅ Extraídas {content_count} páginas")

                    
                    if total_content_extracted > 0:
                        aco_result = {
                            'type': 'aco_completed',
                            'content_extracted': total_content_extracted,
                            'aco_statistics': {'success_rate': 0.8}
                        }
                    else:
                        aco_result = {'type': 'error'}
                else:
                    
                    aco_result = self.crawler_agent.receive({
                        'type': 'search_google_aco',
                        'keywords': problematic_keywords,
                        'improved_query': improved_query,
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
                        
                        print("🔄 Consultando RAG con información obtenida por ACO...")
                        new_response = self.rag_agent.receive({'type': 'query', 'query': query}, self)

                        if new_response['type'] == 'answer':
                            print("✅ Nueva respuesta generada con información de exploración ACO")
                            final_answer = new_response['answer']
                            self.context_agent.receive({
                                'type': 'add_interaction',
                                'query': query,
                                'response': final_answer
                            }, self)
                        else:
                            return new_response.get('msg', 'Error en la nueva consulta después de exploración ACO')
                    else:
                        print("⚠️ ACO no extrajo contenido útil, intentando método alternativo...")
                        
                        return self._fallback_search_method(problematic_keywords, query, improved_query)
                else:
                    print("❌ Error en exploración ACO, intentando método alternativo...")
                    return self._fallback_search_method(problematic_keywords, query, improved_query)

            
            return self._handle_route_suggestion(query, final_answer)

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
            mistral_client = MistralClient(model_name="flash")
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

            response = mistral_client.generate(prompt)
            result = response.lower().strip()

            
            return 'true' in result
        except Exception as e:
            print(f"Error al evaluar la utilidad de la respuesta: {e}")
            
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
            mistral_client = MistralClient(model_name="flash")
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

            response = mistral_client.generate(prompt)
            result = response.strip()

            
            if result.lower() == "ninguna" or not result:
                return []

            
            keywords = [keyword.strip() for keyword in result.split(',') if keyword.strip()]

            return keywords

        except Exception as e:
            print(f"Error al extraer palabras clave problemáticas: {e}")
            return []

    def _fallback_search_method(self, problematic_keywords, query, improved_query):
        """
        Método de fallback cuando ACO falla.
        """
        print("🔄 Ejecutando método de búsqueda alternativo...")

        
        crawl_result = self.crawler_agent.receive({
            'type': 'crawl_keywords',
            'keywords': problematic_keywords,
            'improved_query': improved_query
        }, self)

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

    def _is_direct_route_request(self, query: str) -> bool:
        """Determina si es solicitud directa de ruta"""
        route_keywords = [
            'ruta para visitar',
            'recorrido para visitar',
            'ruta turística',
            'mejor ruta',
            'optimizar visita',
            'recorrido óptimo',
            'orden para visitar',
            'plan de visita',
            'itinerario para',
            'ruta optimizada'
        ]
        return any(kw in query.lower() for kw in route_keywords)

    def _handle_direct_route_request(self, query) -> str:
        """Maneja solicitudes explícitas de rutas"""
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
                return "Lo siento, no pude generar una ruta en este momento."

        
        if places:
            return f"Necesito al menos 2 lugares para generar una ruta. Solo identifiqué: {', '.join(places)}"
        return "Por favor, mencione al menos dos lugares para generar una ruta."

    def _is_route_confirmation(self, query: str) -> bool:
        """Determina si la consulta es una confirmación de ruta basada en contexto"""
        
        last_response = self.context_agent.receive({'type': 'get_last_response'}, self)
        
        if not last_response or "optimice una ruta" not in last_response:
            return False

        
        confirmations = ['sí', 'si', 's', 'yes', 'y', 'por favor', 'claro', 'adelante', 'ok', 'deseo', 'genial']
        return any(conf in query.lower() for conf in confirmations)

    def _generate_route_from_context(self) -> str:
        """Genera ruta basada en los lugares almacenados en contexto"""
        
        places_result = self.context_agent.receive({
            'type': 'get_relevant_places_from_context'
        }, self)

        if places_result['type'] != 'extracted_places' or len(places_result['places']) < 2:
            return "Lo siento, no tengo suficientes lugares para generar una ruta."

        
        route_result = self.route_agent.receive({
            'type': 'optimize_route',
            'places': places_result['places']
        }, self)

        if route_result['type'] == 'route_result':
            return self._format_route(route_result)
        return "Ocurrió un error al generar la ruta optimizada."

    def _handle_route_suggestion(self, query: str, current_answer: str) -> str:
        """Añade sugerencia de ruta si es relevante"""
        
        offer_decision = self.context_agent.receive({
            'type': 'should_offer_route',
            'query': query,
            'response': current_answer
        }, self)

        if offer_decision['type'] != 'route_offer_decision' or not offer_decision['should_offer']:
            return current_answer

        
        extraction_result = self.context_agent.receive({
            'type': 'extract_relevant_places',
            'response': current_answer
        }, self)

        if extraction_result['type'] != 'extracted_places' or len(extraction_result['places']) < 2:
            return current_answer

        
        self.context_agent.receive({
            'type': 'store_relevant_places',
            'places': extraction_result['places']
        }, self)

        
        places_list = ", ".join(extraction_result['places'][:3])
        if len(extraction_result['places']) > 3:
            places_list += f" y {len(extraction_result['places']) - 3} más"

        self.context_agent.receive({'type': 'add_route_to_answer'}, self)
        return (f"{current_answer}\n\n"
                f"📍 He identificado varios lugares en mi respuesta ({places_list}). "
                "¿Desea que optimice una ruta para visitarlos? "
                "Simplemente responda 'sí' para generarla.")


    def _format_route(self, route_result):
        """Formatea los resultados de la ruta usando Gemini para un estilo de guía turístico"""
        if route_result['type'] != 'route_result':
            return "No se pudo generar la ruta."

        try:
            places = route_result['order']
            total_distance = route_result['total_distance_km']

            prompt = f"""
            Eres un guía turístico experto. Describe la siguiente ruta optimizada de manera natural y útil:
            
            Lugares a visitar (en orden):
            {", ".join(places)}
            
            Distancia total: {total_distance} km
            Tiempo estimado caminando: {total_distance/5:.1f} horas
            
            Instrucciones:
            1. Comienza con un saludo entusiasta
            2. Si hay lugares en diferentes ciudades, sugiere dividir la ruta en varios días
            3. Para cada lugar, sugiere un tiempo de visita razonable (ej: 1-2 horas para museos, 2-3 horas para parques grandes)
            4. Incluye consejos prácticos (calzado cómodo, horarios, transporte entre ciudades)
            5. Mantén un tono amigable y motivador
            6. Destaca experiencias únicas en cada lugar
            7. Termina con una recomendación general y buena energía
            
            """

            mistral_client = MistralClient(model_name="flash")
            response = mistral_client.generate(prompt)
            return response.strip()

        except Exception as e:
            
            print(f"Error al generar descripción con Mistral: {e}")
            route_str = "🗺️ **Ruta optimizada**:\n"
            for i, place in enumerate(places):
                route_str += f"{i+1}. {place}\n"
            route_str += f"\n📏 Distancia total: {total_distance} km"
            route_str += f"\n⏱️ Tiempo estimado: {total_distance/5:.1f} horas"
            return route_str

    def _is_vacation_planning_request(self, query: str) -> bool:
        """
        Detecta si el usuario quiere planificar vacaciones
        """
        planning_keywords = [
            'planificar vacaciones',
            'planear vacaciones',
            'organizar viaje',
            'planificar viaje',
            'ayuda con vacaciones',
            'quiero viajar',
            'necesito planificar',
            'guía turístico',
            'planear mi viaje',
            'organizar mis vacaciones'
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in planning_keywords)

    def _start_vacation_planning(self) -> str:
        """
        Inicia el flujo de planificación de vacaciones
        """
        if not self.tourist_guide_agent:
            return "Lo siento, el servicio de planificación de vacaciones no está disponible en este momento."

        
        self.planning_state['mode'] = 'planning'
        self.planning_state['iterations'] = 0
        self.planning_state['aco_depth'] = 1

        
        response = self.tourist_guide_agent.receive({'type': 'start_conversation'}, self)

        if response['type'] == 'guide_response':
            print("🏖️ Modo planificación de vacaciones activado")
            return response['message']
        else:
            self.planning_state['mode'] = 'normal'
            return "Error al iniciar el asistente de planificación."

    def _handle_planning_mode(self, user_message: str) -> str:
        """
        Maneja las interacciones en modo planificación
        """
        
        if self._wants_to_exit_planning(user_message):
            self.planning_state['mode'] = 'normal'
            return "He salido del modo planificación. Ahora puedes hacerme consultas normales sobre turismo."

        
        response = self.tourist_guide_agent.receive({
            'type': 'user_message',
            'message': user_message
        }, self)

        if response['type'] == 'guide_response':
            
            if response.get('preferences_collected', False):
                print("✅ Preferencias recopiladas, iniciando búsqueda con ACO")
                
                final_prefs = response.get('final_preferences')
                if final_prefs:
                    self.planning_state['preferences'] = final_prefs
                    return self._execute_aco_search_with_preferences(final_prefs)
                else:
                    
                    
                    current_prefs = response.get('current_preferences')
                    if current_prefs:
                        self.planning_state['preferences'] = current_prefs
                        return self._execute_aco_search_with_preferences(current_prefs)
                    else:
                        
                        prefs_response = self.tourist_guide_agent.receive({'type': 'get_preferences'}, self)
                        if prefs_response['type'] == 'preferences':
                            self.planning_state['preferences'] = prefs_response['preferences']
                            return self._execute_aco_search_with_preferences(prefs_response['preferences'])
                        else:
                            return "Error al obtener las preferencias. Por favor, intenta de nuevo."
            else:
                return response['message']
        else:
            return "Error procesando tu mensaje. Por favor, intenta de nuevo."

    def _wants_to_exit_planning(self, message: str) -> bool:
        """
        Detecta si el usuario quiere salir del modo planificación
        """
        
        exit_keywords = ['cancelar', 'cancelar planificación', 'salir del modo planificación']
        message_lower = message.lower().strip()

        
        for keyword in exit_keywords:
            if message_lower == keyword or message_lower.startswith(keyword):
                return True

        return False

    def _execute_aco_search_with_preferences(self, preferences: dict) -> str:
        """
        Ejecuta búsqueda con las preferencias recopiladas
        IMPORTANTE: Primero intenta usar la información de la BD local antes de buscar en DuckDuckGo
        """
        
        structured_prefs = self.tourist_guide_agent.get_structured_preferences()

        destination = preferences.get('destination', '')
        interests = preferences.get('interests', [])

        print(f"🎯 Destino: {destination}")
        print(f"🎯 Intereses: {interests}")

        
        print("📚 Consultando información existente en la base de datos...")

        
        itinerary_query = f"Crear itinerario turístico para {destination}"
        if interests:
            itinerary_query += f" incluyendo {', '.join(interests)}"

        
        response = self.rag_agent.receive({'type': 'query', 'query': itinerary_query}, self)

        if response['type'] == 'answer':
            
            evaluation = self._evaluate_response_usefulness(itinerary_query, response['answer'])

            if evaluation:
                
                print("✅ Encontré suficiente información en la base de datos local")
                return self._generate_travel_itinerary(preferences, structured_prefs)
            else:
                
                print("⚠️ La información en la base de datos no es suficiente")
                print("🔍 Iniciando búsqueda en DuckDuckGo para obtener más información...")

                
                search_queries = self._create_specific_search_queries(destination, interests)

                print(f"📋 Se realizarán {len(search_queries)} búsquedas específicas:")
                for i, query in enumerate(search_queries, 1):
                    print(f"   {i}. {query}")

                total_content_extracted = 0

                
                for query in search_queries:
                    print(f"\n🔍 Buscando: '{query}'")

                    
                    aco_result = self.crawler_agent.receive({
                        'type': 'search_google_aco',
                        'keywords': [query],  
                        'improved_query': query,
                        'max_urls': 8,  
                        'max_depth': self.planning_state['aco_depth']
                    }, self)

                    if aco_result.get('type') == 'aco_completed' and aco_result.get('content_extracted'):
                        content_count = aco_result.get('content_extracted', 0)
                        total_content_extracted += content_count
                        print(f"   ✅ Extraídas {content_count} páginas para '{query}'")
                    else:
                        print(f"   ⚠️ No se encontraron resultados para '{query}'")

                if total_content_extracted > 0:
                    print(f"\n✅ Total de páginas extraídas: {total_content_extracted}")

                    
                    self.planning_state['aco_depth'] += 1
                    self.planning_state['iterations'] += 1

                    
                    return self._generate_travel_itinerary(preferences, structured_prefs)
                else:
                    
                    print("⚠️ No se encontró información adicional en DuckDuckGo")
                    print("📚 Generando itinerario con la información disponible en la base de datos...")
                    return self._generate_travel_itinerary(preferences, structured_prefs)
        else:
            
            print("❌ Error al consultar la base de datos, buscando en DuckDuckGo...")

            
            search_queries = self._create_specific_search_queries(destination, interests)

            total_content_extracted = 0

            for query in search_queries:
                aco_result = self.crawler_agent.receive({
                    'type': 'search_google_aco',
                    'keywords': [query],
                    'improved_query': query,
                    'max_urls': 8,
                    'max_depth': self.planning_state['aco_depth']
                }, self)

                if aco_result.get('type') == 'aco_completed' and aco_result.get('content_extracted'):
                    content_count = aco_result.get('content_extracted', 0)
                    total_content_extracted += content_count

            if total_content_extracted > 0:
                return self._generate_travel_itinerary(preferences, structured_prefs)
            else:
                return "Lo siento, no pude encontrar suficiente información para crear tu itinerario. Por favor, intenta con otro destino."

    def _generate_travel_itinerary(self, preferences: dict, structured_prefs: dict) -> str:
        """
        Genera un itinerario de viaje basado en las preferencias y la información recopilada
        IMPORTANTE: Usa información de la BD local y optimiza rutas con RouteAgent
        """
        
        if not preferences:
            preferences = self.planning_state.get('preferences', {})

        if not preferences:
            return "Error: No se encontraron las preferencias del usuario. Por favor, intenta iniciar el proceso de nuevo."

        destination = preferences.get('destination', 'tu destino')
        interests = preferences.get('interests', [])
        duration = preferences.get('duration', 'No especificada')

        
        itinerary_query = f"Crear itinerario turístico para {destination}"
        if interests:
            itinerary_query += f" incluyendo {', '.join(interests)}"

        
        print("📅 Generando itinerario personalizado desde la base de datos...")
        response = self.rag_agent.receive({'type': 'query', 'query': itinerary_query}, self)

        if response['type'] == 'answer':
            
            print("🗺️ Extrayendo lugares para optimizar rutas...")
            extraction_result = self.context_agent.receive({
                'type': 'extract_relevant_places',
                'response': response['answer']
            }, self)

            optimized_routes = {}
            if extraction_result['type'] == 'extracted_places' and len(extraction_result['places']) >= 2:
                places = extraction_result['places']


                
                print("🚀 Optimizando rutas con el agente de rutas...")

                
                days_info = self._estimate_days_needed(len(places), duration)

                if days_info['days'] > 1:
                    
                    places_per_day = self._distribute_places_by_days(places, days_info['days'])

                    for day_num, day_places in enumerate(places_per_day, 1):
                        if len(day_places) >= 2:
                            route_result = self.route_agent.receive({
                                'type': 'optimize_route',
                                'places': day_places
                            }, self)

                            if route_result['type'] == 'route_result':
                                optimized_routes[f'day_{day_num}'] = {
                                    'places': route_result['order'],
                                    'distance_km': route_result['total_distance_km']
                                }
                else:
                    
                    route_result = self.route_agent.receive({
                        'type': 'optimize_route',
                        'places': places
                    }, self)

                    if route_result['type'] == 'route_result':
                        optimized_routes['day_1'] = {
                            'places': route_result['order'],
                            'distance_km': route_result['total_distance_km']
                        }

                print(f"✅ Rutas optimizadas para {len(optimized_routes)} día(s)")

            
            if optimized_routes:
                
                itinerary = self._format_as_itinerary_with_routes(
                    response['answer'],
                    preferences,
                    optimized_routes
                )
            else:
                
                itinerary = self._format_as_itinerary(
                    response['answer'],
                    preferences
                )

            
            self.context_agent.receive({
                'type': 'add_interaction',
                'query': f"Itinerario para {destination}",
                'response': itinerary
            }, self)

            
            self.planning_state['mode'] = 'normal'
            self.planning_state['preferences'] = None
            self.planning_state['aco_depth'] = 1
            self.planning_state['iterations'] = 0

            return itinerary
        else:
            return "No pude generar un itinerario con la información disponible. Por favor, intenta hacer consultas específicas sobre tu destino."

    def _format_as_itinerary(self, raw_response: str, preferences: dict) -> str:
        """
        Formatea la respuesta como un itinerario estructurado
        """

        print("------------------------------\n"+raw_response+"\n------------------------------")
        try:
            mistral_client = MistralClient(model_name="flash")

            prompt = f"""
            Eres un experto planificador de viajes. Transforma la siguiente información en un itinerario de viaje estructurado y atractivo.

            Información disponible:
            {raw_response}

            Preferencias del viajero:
            - Destino: {preferences.get('destination')}
            - Intereses: {', '.join(preferences.get('interests', []))}
            - Duración: {preferences.get('duration', 'No especificada')}
            - Presupuesto: {preferences.get('budget', 'No especificado')}

            INSTRUCCIONES:
            1. Crea un itinerario día por día si es posible
            2. Incluye horarios sugeridos para cada actividad
            3. Agrupa actividades por proximidad geográfica
            4. Incluye recomendaciones de restaurantes para almuerzo y cena
            5. Añade consejos prácticos (transporte, entradas, mejores horarios)
            6. Usa emojis para hacer el itinerario más visual
            7. Mantén un tono entusiasta y personalizado

            Formato deseado:
            🌟 ITINERARIO PARA [DESTINO]

            📅 DÍA 1: [Título del día]
            🕐 Mañana (9:00-12:00): [Actividades]
            🍽️ Almuerzo (12:30-14:00): [Restaurante recomendado]
            🕑 Tarde (14:30-18:00): [Actividades]
            🍽️ Cena (19:30-21:00): [Restaurante recomendado]

            [Continuar con más días si aplica]

            💡 CONSEJOS IMPORTANTES:
            - [Consejo 1]
            - [Consejo 2]

            🎯 RESUMEN:
            [Resumen del itinerario y mensaje motivador]

            IMPORTANTE:
            Responde solo con la información disponible proporcionada.
            Nunca añadas destinos que no aparecen en la información disponible.
            """

            response = mistral_client.generate(prompt)
            formatted_itinerary = response.strip()

            
            simulation_json = format_as_simulation_input(formatted_itinerary, preferences)
            
            
            if self.simulation_agent:
                print("🧩 Enviando itinerario al agente de simulación...")
                print("🧩 JSON para simulación:")

                simulation_result = self._run_simulation(simulation_json)
                
                
                if simulation_result:
                    formatted_itinerary += f"\n\n{simulation_result}"
            else:
                print("⚠️ Agente de simulación no disponible")
            return formatted_itinerary

        except Exception as e:
            print(f"Error formateando itinerario: {e}")
            
            return f"""
🌟 ITINERARIO PARA {preferences.get('destination', 'TU DESTINO').upper()}

{raw_response}

💡 Recomendaciones basadas en tus intereses: {', '.join(preferences.get('interests', []))}

¡Disfruta tu viaje!
"""

    def _add_route_optimization_offer(self, itinerary: str, destination: str) -> str:
        """
        Añade una oferta para optimizar rutas si el itinerario contiene lugares específicos
        """
        
        extraction_result = self.context_agent.receive({
            'type': 'extract_relevant_places',
            'response': itinerary
        }, self)

        if extraction_result['type'] == 'extracted_places' and len(extraction_result['places']) >= 2:
            places_list = ", ".join(extraction_result['places'][:5])
            if len(extraction_result['places']) > 5:
                places_list += f" y {len(extraction_result['places']) - 5} lugares más"

            return f"""{itinerary}

📍 **¿Necesitas optimizar tus rutas?**
He identificado varios lugares en tu itinerario ({places_list}).
Si deseas que optimice las rutas para visitarlos de la manera más eficiente, solo dímelo y crearé rutas optimizadas para cada día de tu viaje.
"""

        return itinerary

    def _detect_user_intent(self, query: str) -> str:
        """
        Detecta la intención del usuario usando Mistral

        Returns:
            'plan_vacation' - Usuario quiere planificar nuevas vacaciones
            'create_itinerary' - Usuario quiere crear itinerario con información actual
            'need_more_info' - Usuario necesita más información sobre un tema
            'normal_query' - Consulta normal del sistema
        """
        try:
            mistral_client = MistralClient(model_name="flash")

            
            context_result = self.context_agent.receive({'type': 'get_context'}, self)
            recent_history = ""
            if context_result['type'] == 'context_data':
                history = context_result.get('history', [])[-3:]  
                for interaction in history:
                    recent_history += f"Usuario: {interaction['query']}\nSistema: {interaction['response'][:200]}...\n\n"

            prompt = f"""
            Analiza la siguiente consulta del usuario y determina su intención principal.

            Contexto de conversación reciente:
            {recent_history}

            Consulta actual del usuario: "{query}"

            INTENCIONES POSIBLES:
            1. 'plan_vacation' - El usuario quiere iniciar la planificación de nuevas vacaciones o un viaje
               Ejemplos: "quiero planificar vacaciones", "ayúdame a organizar un viaje", "necesito planear mis vacaciones"

            2. 'create_itinerary' - El usuario quiere crear un itinerario con la información que ya proporcionó
               Ejemplos: "crea el itinerario", "genera mi plan de viaje", "hazme el itinerario con lo que te dije"

            3. 'need_more_info' - El usuario necesita más información específica sobre algún tema turístico
               Ejemplos: "dime más sobre las playas", "necesito información sobre hoteles", "qué más hay para hacer"

            4. 'normal_query' - Consulta normal sobre turismo que no encaja en las categorías anteriores

            INSTRUCCIONES:
            - Considera el contexto de la conversación para entender mejor la intención
            - Si el usuario ya estaba planificando vacaciones y pide el itinerario, es 'create_itinerary'
            - Si el usuario menciona explícitamente buscar o necesitar más información, es 'need_more_info'

            Responde ÚNICAMENTE con una de estas opciones: 'plan_vacation', 'create_itinerary', 'need_more_info', 'normal_query'
            """

            response = mistral_client.generate(prompt)
            intent = response.strip().lower()

            
            valid_intents = ['plan_vacation', 'create_itinerary', 'need_more_info', 'normal_query']
            if intent in valid_intents:
                return intent

            
            query_lower = query.lower()

            if self._is_vacation_planning_request(query):
                return 'plan_vacation'
            elif any(kw in query_lower for kw in ['crea el itinerario', 'genera el itinerario', 'hazme el itinerario', 'quiero el itinerario']):
                return 'create_itinerary'
            elif any(kw in query_lower for kw in ['más información', 'dime más', 'necesito saber más', 'busca más']):
                return 'need_more_info'
            else:
                return 'normal_query'

        except Exception as e:
            print(f"Error detectando intención: {e}")
            
            return 'normal_query'

    def _create_itinerary_with_current_info(self) -> str:
        """
        Crea un itinerario con la información actual disponible
        """
        
        if self.planning_state.get('preferences'):
            preferences = self.planning_state['preferences']
            print("📋 Creando itinerario con las preferencias guardadas...")
            return self._generate_travel_itinerary(preferences, {})

        
        context_result = self.context_agent.receive({'type': 'get_context'}, self)
        if context_result['type'] == 'context_data':
            
            history = context_result.get('history', [])

            
            preferences = self._extract_preferences_from_history(history)

            if preferences and (preferences.get('destination') or preferences.get('interests')):
                print("📋 Creando itinerario basado en la conversación anterior...")
                return self._generate_travel_itinerary(preferences, {})

        return """No tengo suficiente información para crear un itinerario.

Para crear un itinerario personalizado necesito saber:
- ¿A dónde quieres viajar? (destino)
- ¿Qué te gustaría hacer? (playas, museos, restaurantes, etc.)

Puedes decirme "quiero planificar vacaciones" para iniciar una conversación guiada, o simplemente dime tu destino e intereses."""

    def _search_more_information(self, query: str) -> str:
        """
        Busca más información sobre un tema específico usando ACO
        """
        print("🔍 Detectada necesidad de más información...")

        
        topic_keywords = self._extract_topic_keywords(query)

        if not topic_keywords:
            return "Por favor, especifica sobre qué tema necesitas más información."

        print(f"🐜 Buscando información adicional sobre: {', '.join(topic_keywords)}")

        
        aco_result = self.crawler_agent.receive({
            'type': 'search_google_aco',
            'keywords': topic_keywords,
            'improved_query': query,
            'max_urls': 10,
            'max_depth': 2
        }, self)

        if aco_result.get('type') == 'aco_completed' and aco_result.get('content_extracted'):
            content_count = aco_result.get('content_extracted', 0)
            print(f"✅ Encontré {content_count} fuentes de información adicional")

            
            response = self.rag_agent.receive({'type': 'query', 'query': query}, self)

            if response['type'] == 'answer':
                final_answer = response['answer']

                
                self.context_agent.receive({
                    'type': 'add_interaction',
                    'query': query,
                    'response': final_answer
                }, self)

                return f"📚 He encontrado información adicional:\n\n{final_answer}"

        return "No pude encontrar información adicional sobre ese tema. Por favor, intenta ser más específico."

    def _extract_preferences_from_history(self, history: list) -> dict:
        """
        Extrae preferencias de viaje del historial de conversación
        """
        if not history:
            return {}

        try:
            mistral_client = MistralClient(model_name="flash")

            
            history_text = ""
            for interaction in history[-5:]:  
                history_text += f"Usuario: {interaction['query']}\n"
                history_text += f"Sistema: {interaction['response'][:300]}...\n\n"

            prompt = f"""
            Analiza el siguiente historial de conversación y extrae las preferencias de viaje del usuario.

            Historial:
            {history_text}

            Extrae la siguiente información si está disponible:
            - destination: ciudad o país de destino
            - interests: lista de intereses (playas, museos, restaurantes, etc.)
            - budget: presupuesto mencionado
            - duration: duración del viaje
            - travel_dates: fechas de viaje

            Devuelve un JSON con solo los campos que puedas extraer del historial.
            Si no hay información de viaje, devuelve un JSON vacío {{}}.

            IMPORTANTE: Devuelve SOLO el JSON, sin explicaciones.
            """

            response = mistral_client.generate(prompt)
            json_str = response.strip()

            
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx]
                import json
                return json.loads(json_str)

        except Exception as e:
            print(f"Error extrayendo preferencias del historial: {e}")

        return {}

    def _extract_topic_keywords(self, query: str) -> list:
        """
        Extrae palabras clave del tema sobre el que se necesita más información
        """
        try:
            mistral_client = MistralClient(model_name="flash")

            prompt = f"""
            El usuario necesita más información sobre algo. Extrae las palabras clave del tema específico.

            Consulta del usuario: "{query}"

            INSTRUCCIONES:
            - Identifica el tema principal sobre el que se necesita información
            - Extrae sustantivos, lugares, actividades o conceptos específicos
            - No incluyas palabras como "más", "información", "dime", etc.
            - Enfócate en el QUÉ se quiere saber

            Devuelve SOLO las palabras clave separadas por comas, sin explicaciones.
            Si no hay tema claro, responde con "ninguno".
            """

            response = mistral_client.generate(prompt)
            result = response.strip()

            if result.lower() == "ninguno" or not result:
                return []

            keywords = [kw.strip() for kw in result.split(',') if kw.strip()]
            return keywords

        except Exception as e:
            print(f"Error extrayendo palabras clave del tema: {e}")
            return []

    def _create_specific_search_queries(self, destination: str, interests: List[str]) -> List[str]:
        """
        Crea consultas de búsqueda específicas para cada combinación de destino + interés

        Args:
            destination: Destino del viaje
            interests: Lista de intereses del usuario

        Returns:
            Lista de consultas de búsqueda específicas
        """
        search_queries = []

        
        interest_mapping = {
            'accommodation': ['mejores hoteles', 'alojamiento recomendado', 'donde hospedarse'],
            'hotels': ['mejores hoteles', 'hoteles recomendados', 'alojamiento'],
            'beaches': ['mejores playas', 'playas más bonitas', 'playas turísticas'],
            'museums': ['museos importantes', 'mejores museos', 'museos que visitar'],
            'restaurants': ['mejores restaurantes', 'donde comer', 'gastronomía local'],
            'activities': ['actividades turísticas', 'qué hacer', 'atracciones principales'],
            'shopping': ['centros comerciales', 'donde comprar', 'mejores tiendas'],
            'nightlife': ['vida nocturna', 'bares y discotecas', 'entretenimiento nocturno'],
            'nature': ['parques naturales', 'naturaleza', 'ecoturismo'],
            'culture': ['sitios culturales', 'patrimonio cultural', 'lugares históricos']
        }

        
        if destination:
            for interest in interests:
                
                search_terms = interest_mapping.get(interest.lower(), [interest])

                
                for term in search_terms:
                    query = f"{term} en {destination}"
                    search_queries.append(query)

                
                if interest not in interest_mapping:
                    search_queries.append(f"{interest} en {destination}")

        
        elif interests:
            for interest in interests:
                search_terms = interest_mapping.get(interest.lower(), [interest])
                for term in search_terms:
                    search_queries.append(f"{term} turismo")

        
        if destination:
            search_queries.append(f"guía turística {destination}")
            search_queries.append(f"qué visitar en {destination}")

        
        seen = set()
        unique_queries = []
        for query in search_queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)

        
        return unique_queries[:10]

    def _estimate_days_needed(self, num_places: int, duration_str: str) -> dict:
        """
        Estima el número de días necesarios basado en la cantidad de lugares y duración especificada
        """
        
        import re
        days = 1  

        if duration_str and duration_str != 'No especificada':
            
            numbers = re.findall(r'\d+', str(duration_str).lower())
            if numbers:
                days = int(numbers[0])
            elif 'semana' in duration_str.lower():
                days = 7
            elif 'fin de semana' in duration_str.lower():
                days = 2

        
        if duration_str == 'No especificada':
            
            days = max(1, (num_places + 2) // 3)

        return {'days': days, 'places_per_day': max(1, num_places // days)}

    def _distribute_places_by_days(self, places: List[str], days: int) -> List[List[str]]:
        """
        Distribuye los lugares equitativamente entre los días disponibles
        """
        if days <= 1:
            return [places]

        places_per_day = len(places) // days
        remainder = len(places) % days

        distribution = []
        start_idx = 0

        for day in range(days):
            
            end_idx = start_idx + places_per_day + (1 if day < remainder else 0)
            distribution.append(places[start_idx:end_idx])
            start_idx = end_idx

        return distribution

    def _format_as_itinerary_with_routes(self, raw_response: str, preferences: dict, optimized_routes: dict) -> str:
        """
        Formatea la respuesta como un itinerario estructurado con rutas optimizadas
        """
        try:
            mistral_client = MistralClient(model_name="flash")

            
            routes_info = ""
            for day_key, route_data in optimized_routes.items():
                day_num = day_key.replace('day_', '')
                places_order = " → ".join(route_data['places'])
                distance = route_data['distance_km']
                routes_info += f"\nDía {day_num}: {places_order} (Distancia total: {distance:.1f} km)"

            prompt = f"""
            Eres un experto planificador de viajes. Crea un itinerario de viaje estructurado y atractivo.

            Información disponible de la base de datos:
            {raw_response}

            Rutas optimizadas por día:
            {routes_info}

            Preferencias del viajero:
            - Destino: {preferences.get('destination')}
            - Intereses: {', '.join(preferences.get('interests', []))}
            - Duración: {preferences.get('duration', 'No especificada')}
            - Presupuesto: {preferences.get('budget', 'No especificado')}

            INSTRUCCIONES IMPORTANTES:
            1. USA EXACTAMENTE el orden de lugares proporcionado en las rutas optimizadas
            2. Para cada día, sigue el orden de visita indicado con las flechas (→)
            3. Incluye horarios sugeridos para cada lugar
            4. Añade tiempos de desplazamiento entre lugares basados en las distancias
            5. Incluye recomendaciones de restaurantes para almuerzo y cena
            6. Añade consejos prácticos (transporte, entradas, mejores horarios)
            7. Usa emojis para hacer el itinerario más visual
            8. Mantén un tono entusiasta y personalizado

            Formato deseado:
            🌟 ITINERARIO OPTIMIZADO PARA [DESTINO]

            📅 DÍA 1: [Título descriptivo del día]
            📍 Ruta del día: [Lugar 1] → [Lugar 2] → [Lugar 3]
            📏 Distancia total: X.X km

            🕐 9:00 - [Lugar 1]
            [Descripción y tiempo sugerido de visita]

            🚶 Desplazamiento (X minutos)

            🕑 11:00 - [Lugar 2]
            [Descripción y tiempo sugerido de visita]

            🍽️ 13:00 - Almuerzo en [Restaurante recomendado cerca]

            [Continuar con el resto del día...]

            💡 CONSEJOS DEL DÍA:
            - [Consejo específico para este día]

            [Repetir formato para cada día]

            🎯 RESUMEN GENERAL:
            - Distancia total del viaje: X km
            - Lugares visitados: X
            - [Mensaje motivador final]

            IMPORTANTE: Usa SOLO la información proporcionada. No inventes lugares ni añadas destinos que no aparecen en los datos.
            """

            response = mistral_client.generate(prompt)
            formatted_itinerary = response.strip()

            simulation_json = format_as_simulation_input(formatted_itinerary, preferences)
            
            if self.simulation_agent:
                print("🧩 Enviando itinerario al agente de simulación...")
                print("🧩 JSON para simulación:")
                simulation_result = self._run_simulation(simulation_json)

                
                if simulation_result:
                    formatted_itinerary += f"\n\n{simulation_result}"
            else:
                print("⚠️ Agente de simulación no disponible")
                print("🧩 JSON para simulación:")
                print(json.dumps(simulation_json, ensure_ascii=False, indent=2))
            return formatted_itinerary

        except Exception as e:
            print(f"Error formateando itinerario con rutas: {e}")
            
            fallback = f"🌟 ITINERARIO PARA {preferences.get('destination', 'TU DESTINO').upper()}\n\n"

            for day_key, route_data in optimized_routes.items():
                day_num = day_key.replace('day_', '')
                fallback += f"📅 DÍA {day_num}:\n"
                fallback += f"📍 Ruta optimizada: {' → '.join(route_data['places'])}\n"
                fallback += f"📏 Distancia: {route_data['distance_km']:.1f} km\n\n"

            fallback += f"\n{raw_response}\n\n"
            fallback += f"💡 Recomendaciones basadas en tus intereses: {', '.join(preferences.get('interests', []))}\n\n"
            fallback += "¡Disfruta tu viaje!"

            return fallback

    def _run_simulation(self, simulation_json: dict) -> str:
        """
        Ejecuta la simulación del itinerario usando las utilidades de simulación
        
        Args:
            simulation_json: JSON estructurado con el itinerario para simular
            
        Returns:
            String con los resultados de la simulación formateados
        """
        
        return run_simulation_replicas(self.simulation_agent, simulation_json, num_replicas=30)
