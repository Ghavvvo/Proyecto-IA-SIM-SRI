from autogen import Agent
from typing import List
import google.generativeai as genai

class CoordinatorAgent(Agent):
    def __init__(self, name, crawler_agent, rag_agent, interface_agent, context_agent, route_agent, tourist_guide_agent=None):
        super().__init__(name)
        self.crawler_agent = crawler_agent
        self.rag_agent = rag_agent
        self.interface_agent = interface_agent
        self.context_agent = context_agent
        self.route_agent = route_agent
        self.tourist_guide_agent = tourist_guide_agent
        
        # Estado del flujo de planificación
        self.planning_state = {
            'mode': 'normal',  # 'normal' o 'planning'
            'preferences': None,
            'aco_depth': 1,  # Profundidad inicial para ACO
            'iterations': 0,
            'max_iterations': 5
        }

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
        # PRIMERO: Si estamos en modo planificación, manejar dentro del flujo de planificación
        if self.planning_state['mode'] == 'planning':
            return self._handle_planning_mode(query)
        
        # DESPUÉS: Detectar intención del usuario solo si NO estamos en modo planificación
        user_intent = self._detect_user_intent(query)
        
        # Manejar según la intención detectada
        if user_intent == 'plan_vacation':
            return self._start_vacation_planning()
        
        elif user_intent == 'create_itinerary':
            return self._create_itinerary_with_current_info()
        
        elif user_intent == 'need_more_info':
            return self._search_more_information(query)
        
        # Paso 0: Manejar solicitudes directas de rutas
        if self._is_direct_route_request(query):
            return self._handle_direct_route_request(query)
        
        # Paso 0.5: Verificar si es confirmación de ruta basada en contexto
        if self._is_route_confirmation(query):
            return self._generate_route_from_context()
        
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
                    'improved_query': improved_query,  # Pasar la consulta mejorada
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
                            self.context_agent.receive({
                                'type': 'add_interaction', 
                                'query': query, 
                                'response': final_answer
                            }, self)                            
                        else:
                            return new_response.get('msg', 'Error en la nueva consulta después de exploración ACO')
                    else:
                        print("⚠️ ACO no extrajo contenido útil, intentando método alternativo...")
                        # Fallback a método anterior
                        return self._fallback_search_method(problematic_keywords, query, improved_query)
                else:
                    print("❌ Error en exploración ACO, intentando método alternativo...")
                    return self._fallback_search_method(problematic_keywords, query, improved_query)

            # Manejar sugerencia de ruta si es relevante            
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
    
    def _fallback_search_method(self, problematic_keywords, query, improved_query):
        """
        Método de fallback cuando ACO falla.
        """
        print("🔄 Ejecutando método de búsqueda alternativo...")
        
        # Usar el método anterior como fallback con consulta mejorada
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
        
        # Si no hay suficientes lugares, dar instrucciones claras
        if places:
            return f"Necesito al menos 2 lugares para generar una ruta. Solo identifiqué: {', '.join(places)}"
        return "Por favor, mencione al menos dos lugares para generar una ruta."

    def _is_route_confirmation(self, query: str) -> bool:
        """Determina si la consulta es una confirmación de ruta basada en contexto"""
        # Obtener el último mensaje del sistema
        last_response = self.context_agent.receive({'type': 'get_last_response'}, self)
        # Verificar si el último mensaje contenía una sugerencia de ruta
        if not last_response or "optimice una ruta" not in last_response:
            return False
        
        # Verificar si la consulta actual es una confirmación simple
        confirmations = ['sí', 'si', 's', 'yes', 'y', 'por favor', 'claro', 'adelante', 'ok', 'deseo', 'genial']
        return any(conf in query.lower() for conf in confirmations)

    def _generate_route_from_context(self) -> str:
        """Genera ruta basada en los lugares almacenados en contexto"""
        # Obtener lugares relevantes del contexto
        places_result = self.context_agent.receive({
            'type': 'get_relevant_places_from_context'
        }, self)
        
        if places_result['type'] != 'extracted_places' or len(places_result['places']) < 2:
            return "Lo siento, no tengo suficientes lugares para generar una ruta."
        
        # Generar ruta optimizada
        route_result = self.route_agent.receive({
            'type': 'optimize_route',
            'places': places_result['places']
        }, self)
        
        if route_result['type'] == 'route_result':
            return self._format_route(route_result)
        return "Ocurrió un error al generar la ruta optimizada."

    def _handle_route_suggestion(self, query: str, current_answer: str) -> str:
        """Añade sugerencia de ruta si es relevante"""
        # Detectar si debemos ofrecer ruta
        offer_decision = self.context_agent.receive({
            'type': 'should_offer_route',
            'query': query,
            'response': current_answer
        }, self)
        
        if offer_decision['type'] != 'route_offer_decision' or not offer_decision['should_offer']:
            return current_answer
        
        # Extraer lugares relevantes
        extraction_result = self.context_agent.receive({
            'type': 'extract_relevant_places',
            'response': current_answer
        }, self)
        
        if extraction_result['type'] != 'extracted_places' or len(extraction_result['places']) < 2:
            return current_answer
        
        # Guardar lugares en contexto
        self.context_agent.receive({
            'type': 'store_relevant_places',
            'places': extraction_result['places']
        }, self)
        
        # Añadir invitación
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
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            # Fallback en caso de error
            print(f"Error al generar descripción con Gemini: {e}")
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
        
        # Cambiar a modo planificación
        self.planning_state['mode'] = 'planning'
        self.planning_state['iterations'] = 0
        self.planning_state['aco_depth'] = 1
        
        # Iniciar conversación con el guía turístico
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
        # Verificar si el usuario quiere salir del modo planificación
        if self._wants_to_exit_planning(user_message):
            self.planning_state['mode'] = 'normal'
            return "He salido del modo planificación. Ahora puedes hacerme consultas normales sobre turismo."
        
        # Procesar mensaje con el guía turístico
        response = self.tourist_guide_agent.receive({
            'type': 'user_message',
            'message': user_message
        }, self)
        
        if response['type'] == 'guide_response':
            # Si se completó la recopilación de preferencias
            if response.get('preferences_collected', False):
                print("✅ Preferencias recopiladas, iniciando búsqueda con ACO")
                # Guardar las preferencias en el estado
                final_prefs = response.get('final_preferences')
                if final_prefs:
                    self.planning_state['preferences'] = final_prefs
                    return self._execute_aco_search_with_preferences(final_prefs)
                else:
                    # Si no hay preferencias finales, obtenerlas del agente
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
        # Solo detectar cancelación explícita, no palabras que podrían ser parte de respuestas normales
        exit_keywords = ['cancelar', 'cancelar planificación', 'salir del modo planificación']
        message_lower = message.lower().strip()
        
        # Verificar coincidencia exacta o al inicio de la frase
        for keyword in exit_keywords:
            if message_lower == keyword or message_lower.startswith(keyword):
                return True
        
        return False
    
    def _execute_aco_search_with_preferences(self, preferences: dict) -> str:
        """
        Ejecuta búsqueda ACO con las preferencias recopiladas
        """
        # Obtener palabras clave estructuradas
        structured_prefs = self.tourist_guide_agent.get_structured_preferences()
        
        keywords = structured_prefs['keywords']
        improved_query = structured_prefs['improved_query']
        
        print(f"🐜 Iniciando búsqueda ACO con profundidad {self.planning_state['aco_depth']}")
        print(f"🔍 Palabras clave: {keywords}")
        print(f"📝 Consulta mejorada: {improved_query}")
        
        # Ejecutar búsqueda ACO con profundidad incremental
        aco_result = self.crawler_agent.receive({
            'type': 'search_google_aco',
            'keywords': keywords,
            'improved_query': improved_query,
            'max_urls': 10 + (self.planning_state['iterations'] * 5),  # Incrementar URLs con cada iteración
            'max_depth': self.planning_state['aco_depth']
        }, self)
        
        if aco_result.get('type') == 'aco_completed' and aco_result.get('content_extracted'):
            content_count = aco_result.get('content_extracted', 0)
            print(f"✅ ACO extrajo {content_count} páginas con profundidad {self.planning_state['aco_depth']}")
            
            # Incrementar profundidad para próxima iteración
            self.planning_state['aco_depth'] += 1
            self.planning_state['iterations'] += 1
            
            # Generar itinerario con la información recopilada
            return self._generate_travel_itinerary(preferences, structured_prefs)
        else:
            return "Lo siento, no pude encontrar suficiente información para crear tu itinerario. Por favor, intenta con otro destino."
    
    def _generate_travel_itinerary(self, preferences: dict, structured_prefs: dict) -> str:
        """
        Genera un itinerario de viaje basado en las preferencias y la información recopilada
        """
        # Validar que preferences no sea None
        if not preferences:
            preferences = self.planning_state.get('preferences', {})
        
        if not preferences:
            return "Error: No se encontraron las preferencias del usuario. Por favor, intenta iniciar el proceso de nuevo."
        
        destination = preferences.get('destination', 'tu destino')
        interests = preferences.get('interests', [])
        
        # Construir consulta para el itinerario
        itinerary_query = f"Crear itinerario turístico para {destination}"
        if interests:
            itinerary_query += f" incluyendo {', '.join(interests)}"
        
        # Consultar al RAG con la información actualizada
        print("📅 Generando itinerario personalizado...")
        response = self.rag_agent.receive({'type': 'query', 'query': itinerary_query}, self)
        
        if response['type'] == 'answer':
            # Formatear la respuesta como itinerario
            itinerary = self._format_as_itinerary(response['answer'], preferences)
            
            # Guardar en contexto
            self.context_agent.receive({
                'type': 'add_interaction',
                'query': f"Itinerario para {destination}",
                'response': itinerary
            }, self)
            
            # Resetear estado de planificación
            self.planning_state['mode'] = 'normal'
            self.planning_state['preferences'] = None
            self.planning_state['aco_depth'] = 1
            self.planning_state['iterations'] = 0
            
            # Ofrecer optimización de ruta si hay lugares específicos
            return self._add_route_optimization_offer(itinerary, destination)
        else:
            return "No pude generar un itinerario con la información disponible. Por favor, intenta hacer consultas específicas sobre tu destino."
    
    def _format_as_itinerary(self, raw_response: str, preferences: dict) -> str:
        """
        Formatea la respuesta como un itinerario estructurado
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
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
            """
            
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error formateando itinerario: {e}")
            # Fallback: devolver respuesta con formato básico
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
        # Extraer lugares del itinerario
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
        Detecta la intención del usuario usando Gemini
        
        Returns:
            'plan_vacation' - Usuario quiere planificar nuevas vacaciones
            'create_itinerary' - Usuario quiere crear itinerario con información actual
            'need_more_info' - Usuario necesita más información sobre un tema
            'normal_query' - Consulta normal del sistema
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Obtener contexto de la conversación
            context_result = self.context_agent.receive({'type': 'get_context'}, self)
            recent_history = ""
            if context_result['type'] == 'context_data':
                history = context_result.get('history', [])[-3:]  # Últimas 3 interacciones
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
            
            response = model.generate_content(prompt)
            intent = response.text.strip().lower()
            
            # Validar que la respuesta sea una de las opciones válidas
            valid_intents = ['plan_vacation', 'create_itinerary', 'need_more_info', 'normal_query']
            if intent in valid_intents:
                return intent
            
            # Si no es válida, intentar detectar por palabras clave
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
            # Fallback a detección por palabras clave
            return 'normal_query'
    
    def _create_itinerary_with_current_info(self) -> str:
        """
        Crea un itinerario con la información actual disponible
        """
        # Verificar si tenemos preferencias guardadas
        if self.planning_state.get('preferences'):
            preferences = self.planning_state['preferences']
            print("📋 Creando itinerario con las preferencias guardadas...")
            return self._generate_travel_itinerary(preferences, {})
        
        # Si no hay preferencias guardadas, intentar extraerlas del contexto
        context_result = self.context_agent.receive({'type': 'get_context'}, self)
        if context_result['type'] == 'context_data':
            # Analizar el historial para extraer información de viaje
            history = context_result.get('history', [])
            
            # Usar Gemini para extraer preferencias del historial
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
        
        # Extraer el tema específico de la consulta
        topic_keywords = self._extract_topic_keywords(query)
        
        if not topic_keywords:
            return "Por favor, especifica sobre qué tema necesitas más información."
        
        print(f"🐜 Buscando información adicional sobre: {', '.join(topic_keywords)}")
        
        # Ejecutar búsqueda ACO
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
            
            # Consultar al RAG con la nueva información
            response = self.rag_agent.receive({'type': 'query', 'query': query}, self)
            
            if response['type'] == 'answer':
                final_answer = response['answer']
                
                # Guardar en contexto
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
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Construir el historial como texto
            history_text = ""
            for interaction in history[-5:]:  # Últimas 5 interacciones
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
            
            response = model.generate_content(prompt)
            json_str = response.text.strip()
            
            # Buscar el JSON en la respuesta
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
            model = genai.GenerativeModel('gemini-1.5-flash')
            
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
            
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            if result.lower() == "ninguno" or not result:
                return []
            
            keywords = [kw.strip() for kw in result.split(',') if kw.strip()]
            return keywords
            
        except Exception as e:
            print(f"Error extrayendo palabras clave del tema: {e}")
            return []