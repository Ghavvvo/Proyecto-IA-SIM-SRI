from autogen import Agent
from typing import List, Dict, Any
from core.mistral_config import MistralClient, mistral_generate
from datetime import datetime

class ContextAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history_length = 10  
        self.last_relevant_places = []
        
    def receive(self, message: Dict[str, Any], sender) -> Dict[str, Any]:
        """
        Procesa mensajes del coordinador para manejar el contexto de conversación.
        
        Args:
            message: Diccionario con el tipo de mensaje y datos
            sender: Agente que envía el mensaje
            
        Returns:
            Diccionario con la respuesta procesada
        """
        if message['type'] == 'analyze_query':
            return self._analyze_and_improve_query(message['query'])
        elif message['type'] == 'add_interaction':
            return self._add_interaction(message['query'], message['response'])
        elif message['type'] == 'get_context':
            return self._get_conversation_context()
        elif message['type'] == 'clear_context':
            return self._clear_context()
        elif message['type'] == 'should_offer_route':
            return self._should_offer_route(message['query'], message['response'])       
        elif message['type'] == 'extract_relevant_places':
            return self._extract_relevant_places(message['response'])
        elif message['type'] == 'store_relevant_places':
            self.last_relevant_places = message['places']
            return {'type': 'places_stored'}
        elif message['type'] == 'get_relevant_places_from_context':
            return {'type': 'extracted_places', 'places': self.last_relevant_places}
        elif message['type'] == 'get_last_response':
            return self._get_last_system_response()
        elif message['type'] == 'add_route_to_answer':
            self.add_route_to_answer()
            return{'type': 'route_added'}
        else:
            return {'type': 'error', 'msg': 'Unknown message type'}
    
    def _analyze_and_improve_query(self, query: str) -> Dict[str, Any]:
        """
        Analiza la consulta del usuario en el contexto de la conversación
        y genera una consulta mejorada para el sistema RAG.
        
        Args:
            query: Consulta original del usuario
            
        Returns:
            Diccionario con la consulta mejorada y análisis del contexto
        """
        try:
            
            context_summary = self._build_context_summary()
            
            
            print(f"🔍 Contexto disponible: {context_summary[:100]}...")
            
            
            mistral_client = MistralClient(model_name="flash")
            
            
            prompt = f"""
Eres un experto en mejora de consultas para sistemas RAG. Tu tarea es SIEMPRE mejorar la consulta del usuario basándote en el contexto conversacional.

CONTEXTO PREVIO:
{context_summary}

CONSULTA ORIGINAL: "{query}"

INSTRUCCIONES CRÍTICAS:
1. SIEMPRE debes generar una consulta mejorada, incluso si es solo ligeramente diferente
2. Si hay contexto previo, úsalo para hacer la consulta más específica
3. Si no hay contexto, mejora la consulta con el formato óptimo para una búsqueda en DuckDuckGo
4. Sé conciso, no añadas elementos que no aparecen en la consulta original
5. Asegúrate de que la consulta mejorada contenga toda la información necesaria para una búsqueda efectiva sin contexto adicional
EJEMPLOS DE MEJORAS:
- "¿Qué restaurantes hay en Cuba que me puedan interesar para pasar mis vacaciones?" → "¿Restaurantes en Cuba?"
- "¿Y hoteles?" → "¿Hoteles en [ubicación del contexto]?"

RESPONDE SOLO CON LA CONSULTA MEJORADA, SIN EXPLICACIONES ADICIONALES:
"""
            
            response = mistral_client.generate(prompt)
            improved_query = response.strip().strip('"\'')
            
            
            print(f"🤖 Respuesta de Mistral: {improved_query}")
            
            
            if not improved_query or improved_query.lower() == query.lower() or len(improved_query) < 5:
                
                improved_query = self._apply_basic_improvements(query, context_summary)
                print(f"🔧 Aplicando mejoras básicas: {improved_query}")
            
            
            is_continuation = self._is_query_continuation(query, context_summary)
            
            return {
                'type': 'query_analyzed',
                'analysis': {
                    'improved_query': improved_query,
                    'context_analysis': {
                        'is_continuation': is_continuation,
                        'related_topics': self._extract_topics_from_context(),
                        'user_intent': self._determine_user_intent(query, improved_query),
                        'context_relevance': 'alta' if is_continuation else 'media'
                    },
                    'improvements_made': self._identify_improvements(query, improved_query),
                    'original_query': query
                }
            }
                
        except Exception as e:
            print(f"❌ Error al analizar consulta con contexto: {e}")
            
            improved_query = self._apply_basic_improvements(query, self._build_context_summary())
            
            return {
                'type': 'query_analyzed',
                'analysis': {
                    'improved_query': improved_query,
                    'context_analysis': {
                        'is_continuation': len(self.conversation_history) > 0,
                        'related_topics': [],
                        'user_intent': 'Consulta con mejoras básicas',
                        'context_relevance': 'baja'
                    },
                    'improvements_made': ['Mejoras básicas aplicadas'],
                    'original_query': query
                }
            }
    
    def _add_interaction(self, query: str, response: str) -> Dict[str, Any]:
        """
        Añade una nueva interacción al historial de conversación.
        
        Args:
            query: Consulta del usuario
            response: Respuesta del sistema
            
        Returns:
            Confirmación de que la interacción fue añadida
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'interaction_id': len(self.conversation_history) + 1
        }
        
        self.conversation_history.append(interaction)
        
        
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return {
            'type': 'interaction_added',
            'interaction_count': len(self.conversation_history)
        }
    
    def _get_conversation_context(self) -> Dict[str, Any]:
        """
        Obtiene el contexto actual de la conversación.
        
        Returns:
            Diccionario con el contexto de conversación
        """
        return {
            'type': 'context_data',
            'history': self.conversation_history,
            'interaction_count': len(self.conversation_history),
            'context_summary': self._build_context_summary()
        }
    
    def _clear_context(self) -> Dict[str, Any]:
        """
        Limpia el historial de conversación.
        
        Returns:
            Confirmación de que el contexto fue limpiado
        """
        self.conversation_history.clear()
        return {
            'type': 'context_cleared',
            'message': 'Historial de conversación limpiado'
        }
    
    def _build_context_summary(self) -> str:
        """
        Construye un resumen del contexto de conversación.
        
        Returns:
            String con el resumen del contexto
        """
        if not self.conversation_history:
            return "No hay conversación previa."
        
        summary_parts = []
        
        for i, interaction in enumerate(self.conversation_history[-5:], 1):  
            summary_parts.append(f"""
            Interacción {i}:
            - Usuario preguntó: {interaction['query']}
            - Sistema respondió: {interaction['response'][:200]}{'...' if len(interaction['response']) > 200 else ''}
            """)
        
        return "\n".join(summary_parts)
    
    def _extract_improved_query_from_text(self, text: str, original_query: str) -> str:
        """
        Extrae la consulta mejorada del texto de respuesta cuando no es JSON válido.
        
        Args:
            text: Texto de respuesta de Mistral
            original_query: Consulta original como fallback
            
        Returns:
            Consulta mejorada extraída o la original si no se puede extraer
        """
        try:
            
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['consulta mejorada', 'improved query', 'mejor consulta']):
                    
                    if ':' in line:
                        improved = line.split(':', 1)[1].strip()
                        if improved and len(improved) > 10:  
                            return improved.strip('"\'')
            
            
            potential_queries = [line.strip() for line in lines if len(line.strip()) > 20 and '?' in line]
            if potential_queries:
                return potential_queries[0].strip('"\'')
                
        except Exception as e:
            print(f"Error extrayendo consulta mejorada: {e}")
        
        return original_query
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la conversación actual.
        
        Returns:
            Diccionario con estadísticas de conversación
        """
        if not self.conversation_history:
            return {
                'total_interactions': 0,
                'average_query_length': 0,
                'average_response_length': 0,
                'most_recent_topic': None
            }
        
        total_interactions = len(self.conversation_history)
        avg_query_length = sum(len(interaction['query']) for interaction in self.conversation_history) / total_interactions
        avg_response_length = sum(len(interaction['response']) for interaction in self.conversation_history) / total_interactions
        
        return {
            'total_interactions': total_interactions,
            'average_query_length': round(avg_query_length, 2),
            'average_response_length': round(avg_response_length, 2),
            'most_recent_topic': self.conversation_history[-1]['query'] if self.conversation_history else None,
            'conversation_duration': self._calculate_conversation_duration()
        }
    
    def _calculate_conversation_duration(self) -> str:
        """
        Calcula la duración de la conversación.
        
        Returns:
            String con la duración de la conversación
        """
        if len(self.conversation_history) < 2:
            return "Conversación recién iniciada"
        
        try:
            first_interaction = datetime.fromisoformat(self.conversation_history[0]['timestamp'])
            last_interaction = datetime.fromisoformat(self.conversation_history[-1]['timestamp'])
            duration = last_interaction - first_interaction
            
            if duration.total_seconds() < 60:
                return f"{int(duration.total_seconds())} segundos"
            elif duration.total_seconds() < 3600:
                return f"{int(duration.total_seconds() / 60)} minutos"
            else:
                return f"{int(duration.total_seconds() / 3600)} horas"
                
        except Exception:
            return "Duración no disponible"
    
    def _apply_basic_improvements(self, query: str, context_summary: str) -> str:
        """
        Aplica mejoras básicas a la consulta cuando Mistral no está disponible o falla.
        
        Args:
            query: Consulta original
            context_summary: Resumen del contexto
            
        Returns:
            Consulta con mejoras básicas aplicadas
        """
        improved_query = query
        
        
        locations = self._extract_locations_from_context(context_summary)
        
        
        if query.lower().startswith(('¿y ', '¿qué ', 'y ', 'qué ')):
            
            if locations:
                location = locations[0]
                if 'restaurante' in query.lower():
                    improved_query = f"¿Qué restaurantes recomendados hay en {location}?"
                elif 'hotel' in query.lower():
                    improved_query = f"¿Qué hoteles recomendados hay en {location}?"
                elif 'lugar' in query.lower() or 'sitio' in query.lower():
                    improved_query = f"¿Qué lugares turísticos recomendados hay en {location}?"
                else:
                    improved_query = f"{query} en {location}"
        
        
        if 'restaurante' in improved_query.lower() and 'recomendado' not in improved_query.lower():
            improved_query = improved_query.replace('restaurante', 'restaurante recomendado')
        
        if 'hotel' in improved_query.lower() and 'recomendado' not in improved_query.lower():
            improved_query = improved_query.replace('hotel', 'hotel recomendado')
            
        if 'lugar' in improved_query.lower() and 'turístico' not in improved_query.lower():
            improved_query = improved_query.replace('lugar', 'lugar turístico')
        
        
        if len(query.split()) <= 3:
            if 'clima' in query.lower():
                improved_query = f"{query} y recomendaciones de vestimenta"
            elif 'precio' in query.lower():
                improved_query = f"{query} y opciones económicas"
        
        return improved_query if improved_query != query else f"{query} con información detallada"
    
    def _extract_locations_from_context(self, context_summary: str) -> List[str]:
        """
        Extrae ubicaciones mencionadas en el contexto.
        
        Args:
            context_summary: Resumen del contexto
            
        Returns:
            Lista de ubicaciones encontradas
        """
        
        common_locations = [
            'Lima', 'Cusco', 'Arequipa', 'Trujillo', 'Piura', 'Iquitos', 
            'Huancayo', 'Chiclayo', 'Ayacucho', 'Cajamarca', 'Puno',
            'Miraflores', 'Barranco', 'San Isidro', 'Machu Picchu',
            'Valle Sagrado', 'Ollantaytambo', 'Pisac'
        ]
        
        locations = []
        context_lower = context_summary.lower()
        
        for location in common_locations:
            if location.lower() in context_lower:
                locations.append(location)
        
        return locations
    
    def _is_query_continuation(self, query: str, context_summary: str) -> bool:
        """
        Determina si la consulta es una continuación de un tema previo.
        
        Args:
            query: Consulta actual
            context_summary: Resumen del contexto
            
        Returns:
            True si es una continuación, False si es un tema nuevo
        """
        
        continuation_indicators = [
            '¿y ', 'y ', '¿qué más', 'también', 'además', 'otra', 'otro',
            '¿dónde más', '¿cuál', '¿cuáles', 'mejor', 'recomiendan'
        ]
        
        query_lower = query.lower()
        
        
        for indicator in continuation_indicators:
            if query_lower.startswith(indicator):
                return True
        
        
        if not context_summary or context_summary == "No hay conversación previa.":
            return False
        
        
        if len(self.conversation_history) > 0:
            last_query = self.conversation_history[-1]['query'].lower()
            
            
            related_themes = {
                'turismo': ['lugar', 'sitio', 'visitar', 'turístico', 'atracción'],
                'comida': ['restaurante', 'comida', 'comer', 'gastronomía', 'plato'],
                'alojamiento': ['hotel', 'hostal', 'alojamiento', 'dormir', 'hospedaje'],
                'transporte': ['transporte', 'bus', 'taxi', 'avión', 'tren'],
                'clima': ['clima', 'tiempo', 'temperatura', 'lluvia', 'sol']
            }
            
            for theme, keywords in related_themes.items():
                last_has_theme = any(keyword in last_query for keyword in keywords)
                current_has_theme = any(keyword in query_lower for keyword in keywords)
                
                if last_has_theme and current_has_theme:
                    return True
        
        return False
    
    def _extract_topics_from_context(self) -> List[str]:
        """
        Extrae temas principales del contexto de conversación.
        
        Returns:
            Lista de temas identificados
        """
        if not self.conversation_history:
            return []
        
        topics = set()
        
        for interaction in self.conversation_history[-3:]:  
            query = interaction['query'].lower()
            
            
            if any(word in query for word in ['restaurante', 'comida', 'comer', 'gastronomía']):
                topics.add('gastronomía')
            
            if any(word in query for word in ['hotel', 'hostal', 'alojamiento', 'dormir']):
                topics.add('alojamiento')
            
            if any(word in query for word in ['lugar', 'sitio', 'visitar', 'turístico']):
                topics.add('turismo')
            
            if any(word in query for word in ['clima', 'tiempo', 'temperatura']):
                topics.add('clima')
            
            if any(word in query for word in ['transporte', 'bus', 'taxi', 'avión']):
                topics.add('transporte')
            
            
            locations = self._extract_locations_from_context(query)
            topics.update(locations)
        
        return list(topics)
    
    def _determine_user_intent(self, original_query: str, improved_query: str) -> str:
        """
        Determina la intención del usuario basándose en las consultas.
        
        Args:
            original_query: Consulta original
            improved_query: Consulta mejorada
            
        Returns:
            Descripción de la intención del usuario
        """
        query_lower = original_query.lower()
        
        if any(word in query_lower for word in ['¿dónde', 'dónde', 'ubicación', 'dirección']):
            return 'Búsqueda de ubicación'
        
        if any(word in query_lower for word in ['¿qué', 'qué', 'cuál', 'cuáles']):
            if 'restaurante' in query_lower:
                return 'Búsqueda de restaurantes'
            elif 'hotel' in query_lower:
                return 'Búsqueda de alojamiento'
            elif 'lugar' in query_lower:
                return 'Búsqueda de lugares turísticos'
            else:
                return 'Búsqueda de información general'
        
        if any(word in query_lower for word in ['¿cómo', 'cómo']):
            return 'Búsqueda de instrucciones o procedimientos'
        
        if any(word in query_lower for word in ['¿cuánto', 'cuánto', 'precio', 'costo']):
            return 'Consulta de precios'
        
        if any(word in query_lower for word in ['clima', 'tiempo', 'temperatura']):
            return 'Consulta meteorológica'
        
        return 'Consulta informativa general'
    
    def _identify_improvements(self, original_query: str, improved_query: str) -> List[str]:
        """
        Identifica qué mejoras se aplicaron a la consulta.
        
        Args:
            original_query: Consulta original
            improved_query: Consulta mejorada
            
        Returns:
            Lista de mejoras aplicadas
        """
        improvements = []
        
        if len(improved_query) > len(original_query):
            improvements.append('Consulta expandida con más detalles')
        
        if improved_query != original_query:
            improvements.append('Consulta contextualizada')
        
        
        original_lower = original_query.lower()
        improved_lower = improved_query.lower()
        if 'recomendado' in improved_lower and 'recomendado' not in original_lower:
            improvements.append('Añadido filtro de recomendaciones')
        
        locations = self._extract_locations_from_context(improved_query)
        if locations and not any(loc.lower() in original_lower for loc in locations):
            improvements.append('Añadido contexto geográfico')
        
        if any(word in improved_lower for word in ['detallada', 'específica', 'completa']) and \
           not any(word in original_lower for word in ['detallada', 'específica', 'completa']):
            improvements.append('Solicitud de información más específica')
        
        if not improvements:
            improvements.append('Consulta procesada')
        
        return improvements
    
    def _should_offer_route(self, query: str, response: str) -> Dict[str, Any]:
        """
        Determina si se debe ofrecer generar una ruta basado en la consulta y respuesta.
        
        Args:
            query: Consulta original del usuario
            response: Respuesta generada por el sistema
            
        Returns:
            Dict con tipo y booleano indicando si se debe ofrecer ruta
        """
        
        try:
            mistral_client = MistralClient(model_name="flash")
            prompt = f"""
        Eres un experto en análisis de conversaciones para un sistema de guía turístico. 
        Determina si la siguiente respuesta del sistema a una consulta del usuario contiene una lista de lugares de interés que podrían ser visitados en una ruta turística.

        CONSULTA DEL USUARIO:
        {query}

        RESPUESTA DEL SISTEMA:
        {response}

        INSTRUCCIONES:
        - Responde SOLO con 'true' o 'false' (en minúsculas) sin comillas ni explicaciones.
        - Responde 'true' si la respuesta contiene una lista de lugares (mínimo 2) que un turista podría visitar en un recorrido.
        - También responde 'true' si la consulta del usuario explícitamente pidió una ruta, itinerario u orden de visita.
        - Responde 'false' si la respuesta no menciona lugares o si solo menciona un lugar.
        - Responde 'false' si la respuesta es una negativa (ej: "no encontré información") o si no es relevante para planificar una visita.
            """
            result = mistral_client.generate(prompt)
            decision = result.strip().lower()
            should_offer = (decision == 'true')
            
            return {
                'type': 'route_offer_decision',
                'should_offer': should_offer
            }
        except Exception as e:
            print(f"Error al determinar oferta de ruta: {e}")
            
            basic_decision = self._basic_should_offer_route(query, response)
            return {
                'type': 'route_offer_decision',
                'should_offer': basic_decision
            }

    def _basic_should_offer_route(self, query: str, response: str) -> bool:
        """Heurística básica para ofrecer ruta (fallback)"""
        route_keywords = ['ruta', 'recorrido', 'itinerario', 'orden de visita', 'visitar en orden']
        if any(keyword in query.lower() for keyword in route_keywords):
            return True
        
        place_indicators = ['lugares:', 'sitios:', 'puntos de interés:', 'recomendaciones:', 'atracciones:']
        if any(indicator in response.lower() for indicator in place_indicators):
            return True
            
        markers = ['- ', '* ', '• ', '1.', '2.', '3.', '4.', '5.']
        lines = response.split('\n')
        count = 0
        
        for line in lines:
            if any(line.startswith(marker) for marker in markers):
                count += 1
                if count > 2: 
                    return True
        return False

    def _extract_relevant_places(self, response: str) -> Dict[str, Any]:
        """
        Extrae los lugares relevantes de una respuesta usando Mistral.
        
        Args:
            response: Respuesta del sistema
            
        Returns:
            Dict con lista de lugares relevantes
        """
        try:
            mistral_client = MistralClient(model_name="flash")
            prompt = f"""
        Eres un experto en extracción de lugares turísticos. Extrae SOLO los nombres de los lugares turísticos relevantes mencionados en el siguiente texto. 

        INSTRUCCIONES CRÍTICAS:
        - Para CADA lugar turístico, crea UNA SOLA CADENA COMPLETA que incluya:
        1- El nombre principal del lugar
        2- Toda la información contextual disponible (ciudad, región, país) asociada específicamente a ESE lugar
        - Separa CADA lugar completo con punto y coma (;)
        - NUNCA separes el nombre de un lugar de su ubicación con punto y coma
        - Si un lugar no tiene ubicación explícita, inclúyelo solo con su nombre
        - Ignora lugares mencionados incidentalmente (ejemplos o contexto histórico)
        - Si un lugar se repite, inclúyelo solo una vez
        - Devuelve SOLO la lista sin numeración, encabezados ni texto adicional
        - Si no hay lugares, devuelve cadena vacía

        FORMATO CORRECTO:
        "Lugar 1, Ciudad, País; Lugar 2, Ciudad; Lugar 3"

        TEXTO:
        {response}

        LISTA DE LUGARES:
        """
            result = mistral_client.generate(prompt)
            places_str = result.strip()
            
            
            if not places_str:
                return {'type': 'extracted_places', 'places': []}
                
            
            places = []
            seen = set()
            for place in places_str.split(';'):
                cleaned_place = place.strip()
                if cleaned_place and cleaned_place not in seen:
                    seen.add(cleaned_place)
                    places.append(cleaned_place)
                    
            return {
                'type': 'extracted_places',
                'places': places
            }
        except Exception as e:
            print(f"Error al extraer lugares: {e}")
            return {
                'type': 'extracted_places',
                'places': []
            }

    def _get_last_system_response(self) -> str:
        if self.conversation_history:
            return self.conversation_history[-1]['response']
        return ""
    
    def add_route_to_answer(self) -> Dict[str, Any]:
        if not self.conversation_history:
            return {'type': 'error', 'msg': 'No hay conversación para añadir oferta de ruta'}
        
        last_entry = self.conversation_history[-1]
        last_response = last_entry['response']
        
        route_offer = (
            "\n\n¿Desea que optimice una ruta para visitarlos? "
            "Simplemente responda 'sí' para generarla."
        )
        
        if route_offer not in last_response:
            last_entry['response'] += route_offer
            return {
                'type': 'route_offer_added', 
                'message': 'Oferta de ruta añadida a la última respuesta.'
            }
        else:
            return {
                'type': 'route_offer_already_present', 
                'message': 'La oferta de ruta ya estaba presente.'
            }