"""
Agente Guía Turístico - Especializado en recopilar preferencias de viaje
"""

from autogen import Agent
from core.mistral_config import MistralClient, mistral_json
import json
from typing import Dict, List, Optional
from datetime import datetime
import os

class TouristGuideAgent(Agent):
    """
    Agente especializado en actuar como guía turístico para recopilar
    preferencias de viaje del usuario de manera conversacional
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        
        
        self.mistral_client = MistralClient(model_name="flash")
        
        
        self.conversation_state = {
            'phase': 'greeting',  
            'preferences': {
                'destination': None,
                'interests': [],
                'accommodation_type': None,
                'budget': None,
                'duration': None,
                'travel_dates': None,
                'special_requirements': [],
                'preferred_activities': []
            },
            'conversation_history': [],
            'questions_asked': set(),
            'extraction_attempts': 0
        }
        
        
        self.conversation_templates = {
            'greeting': """
            Eres un guía turístico experto y amigable. Tu objetivo es ayudar al usuario a planificar sus vacaciones perfectas.
            
            INSTRUCCIONES:
            1. Saluda cálidamente al usuario
            2. Preséntate como su guía turístico personal
            3. Pregunta sobre su destino de viaje deseado
            4. Mantén un tono entusiasta y profesional
            
            Responde de manera natural y conversacional, en máximo 3-4 líneas.
            """,
            
            'destination': """
            El usuario está interesado en viajar. Necesitas obtener información sobre su destino.
            
            Contexto de la conversación:
            {conversation_history}
            
            Mensaje del usuario: {user_message}
            
            INSTRUCCIONES:
            1. Si el usuario mencionó un destino, confírmalo con entusiasmo
            2. Si no está claro, pide aclaración de manera amigable
            3. Una vez confirmado el destino, pregunta sobre sus intereses principales (playas, museos, restaurantes, etc.)
            4. Mantén la conversación fluida y natural
            
            Responde en máximo 3-4 líneas.
            """,
            
            'interests': """
            Estás ayudando al usuario a planificar un viaje a {destination}.
            
            Contexto de la conversación:
            {conversation_history}
            
            Mensaje del usuario: {user_message}
            
            INSTRUCCIONES:
            1. Identifica los intereses mencionados (hoteles, playas, museos, restaurantes, actividades)
            2. Muestra entusiasmo por sus elecciones
            3. Sugiere categorías adicionales que podrían interesarle
            4. Pregunta sobre preferencias específicas (tipo de hotel, presupuesto, duración del viaje)
            
            Responde de manera conversacional en máximo 4-5 líneas.
            """,
            
            'details': """
            Estás refinando los detalles del viaje a {destination} con intereses en: {interests}.
            
            Contexto de la conversación:
            {conversation_history}
            
            Mensaje del usuario: {user_message}
            
            INSTRUCCIONES:
            1. Recopila detalles específicos que falten (fechas, presupuesto, tipo de alojamiento, duración)
            2. No hagas todas las preguntas a la vez
            3. Prioriza según lo que sea más relevante para la conversación
            4. Si ya tienes suficiente información, prepárate para resumir
            
            Responde de manera natural en máximo 3-4 líneas.
            """,
            
            'summary': """
            Has recopilado las preferencias de viaje del usuario.
            
            Preferencias recopiladas:
            {preferences}
            
            INSTRUCCIONES:
            1. Resume las preferencias del usuario de manera clara y entusiasta
            2. Confirma que la información es correcta
            3. Menciona que ahora buscarás la mejor información para su viaje
            4. Transmite emoción por ayudarle a planificar este viaje
            
            Responde en máximo 4-5 líneas.
            """
        }
        
        
        self.keywords_mapping = {
            'accommodation': ['hotel', 'hostal', 'airbnb', 'alojamiento', 'hospedaje', 'resort', 'apartamento'],
            'beaches': ['playa', 'playas', 'costa', 'mar', 'beach', 'beaches', 'litoral'],
            'museums': ['museo', 'museos', 'museum', 'galería', 'arte', 'historia', 'cultural'],
            'restaurants': ['restaurante', 'restaurantes', 'comida', 'gastronomía', 'comer', 'cocina', 'food'],
            'activities': ['actividad', 'actividades', 'tour', 'excursión', 'aventura', 'deporte', 'entretenimiento'],
            'shopping': ['compras', 'shopping', 'tiendas', 'mercado', 'mall', 'centro comercial'],
            'nightlife': ['vida nocturna', 'bares', 'discoteca', 'club', 'fiesta', 'noche'],
            'nature': ['naturaleza', 'parque', 'montaña', 'senderismo', 'hiking', 'reserva', 'bosque']
        }
    
    def receive(self, message: Dict, sender) -> Dict:
        """
        Procesa mensajes entrantes
        """
        msg_type = message.get('type')
        
        if msg_type == 'start_conversation':
            return self._start_conversation()
        
        elif msg_type == 'user_message':
            user_message = message.get('message', '')
            return self._process_user_message(user_message)
        
        elif msg_type == 'get_preferences':
            return {
                'type': 'preferences',
                'preferences': self.conversation_state['preferences'],
                'conversation_complete': self._is_conversation_complete()
            }
        
        elif msg_type == 'reset':
            return self._reset_conversation()
        
        return {'type': 'error', 'message': 'Tipo de mensaje no reconocido'}
    
    def _start_conversation(self) -> Dict:
        """
        Inicia la conversación con el usuario
        """
        try:
            prompt = self.conversation_templates['greeting']
            response = self.mistral_client.generate(prompt)
            
            greeting = response.strip()
            
            
            self.conversation_state['phase'] = 'destination'
            self.conversation_state['conversation_history'].append({
                'role': 'assistant',
                'content': greeting,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'type': 'guide_response',
                'message': greeting,
                'phase': 'destination',
                'preferences_collected': False
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Error al iniciar conversación: {str(e)}'
            }
    
    def _process_user_message(self, user_message: str) -> Dict:
        """
        Procesa el mensaje del usuario según la fase de la conversación
        """
        try:
            
            self.conversation_state['conversation_history'].append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })
            
            
            
            
            self._extract_preferences(user_message)
            
            
            if self._wants_to_proceed_with_current_info(user_message):
                
                self._user_wants_to_proceed = True
                
                
                if self._has_minimum_preferences():
                    self.conversation_state['phase'] = 'summary'
                    return self._handle_summary_phase(user_message)
                else:
                    
                    return self._explain_missing_info()
            
            
            current_phase = self.conversation_state['phase']
            
            
            if current_phase == 'destination':
                response = self._handle_destination_phase(user_message)
            elif current_phase == 'interests':
                response = self._handle_interests_phase(user_message)
            elif current_phase == 'details':
                response = self._handle_details_phase(user_message)
            elif current_phase == 'summary':
                response = self._handle_summary_phase(user_message)
            else:
                response = self._handle_general_conversation(user_message)
            
            return response
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Error procesando mensaje: {str(e)}'
            }
    
    def _extract_preferences(self, user_message: str) -> None:
        """
        Extrae preferencias del mensaje del usuario usando Mistral
        """
        try:
            extraction_prompt = f"""
            Analiza el siguiente mensaje y extrae información sobre preferencias de viaje.
            
            Mensaje: "{user_message}"
            
            Preferencias actuales:
            {json.dumps(self.conversation_state['preferences'], indent=2)}
            
            IMPORTANTE: Extrae TODA la información de preferencias mencionada en el mensaje, 
            incluso si el usuario también indica que no quiere dar más información.
            
            Por ejemplo:
            - "Solo me interesan museos, nada más" → Extraer "museos" como interés
            - "Quiero ir a playas, eso es todo" → Extraer "playas" como interés
            - "Cuba, ya no preguntes más" → Extraer "Cuba" como destino
            
            Campos posibles:
            - destination: ciudad o país de destino
            - interests: lista de intereses (playas, museos, restaurantes, etc.)
            - accommodation_type: tipo de alojamiento preferido
            - budget: presupuesto aproximado
            - duration: duración del viaje
            - travel_dates: fechas de viaje
            - special_requirements: requisitos especiales
            - preferred_activities: actividades específicas mencionadas
            
            Si no hay información nueva para extraer, devuelve un JSON vacío {{}}.
            
            IMPORTANTE: Devuelve SOLO el JSON, sin explicaciones adicionales.
            """
            
            
            extracted_data = mistral_json(extraction_prompt)
            
            if extracted_data:
                
                for key, value in extracted_data.items():
                    if key in self.conversation_state['preferences']:
                        if isinstance(self.conversation_state['preferences'][key], list):
                            
                            if isinstance(value, list):
                                self.conversation_state['preferences'][key].extend(value)
                            else:
                                self.conversation_state['preferences'][key].append(value)
                            
                            self.conversation_state['preferences'][key] = list(set(self.conversation_state['preferences'][key]))
                        else:
                            
                            self.conversation_state['preferences'][key] = value
            else:
                
                self._manual_extraction(user_message)
                
        except Exception as e:
            print(f"Error en extracción de preferencias: {e}")
            self._manual_extraction(user_message)
    
    def _manual_extraction(self, user_message: str) -> None:
        """
        Extracción manual de respaldo para preferencias básicas
        """
        message_lower = user_message.lower()
        
        
        for category, keywords in self.keywords_mapping.items():
            for keyword in keywords:
                if keyword in message_lower:
                    if category not in self.conversation_state['preferences']['interests']:
                        self.conversation_state['preferences']['interests'].append(category)
    
    def _handle_destination_phase(self, user_message: str) -> Dict:
        """
        Maneja la fase de selección de destino
        """
        
        if self.conversation_state['preferences']['destination']:
            self.conversation_state['phase'] = 'interests'
        
        
        prompt = self.conversation_templates['destination'].format(
            conversation_history=self._format_conversation_history(),
            user_message=user_message
        )
        
        response = self.mistral_client.generate(prompt)
        guide_response = response.strip()
        
        
        self.conversation_state['conversation_history'].append({
            'role': 'assistant',
            'content': guide_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'type': 'guide_response',
            'message': guide_response,
            'phase': self.conversation_state['phase'],
            'preferences_collected': False,
            'current_preferences': self.conversation_state['preferences']
        }
    
    def _handle_interests_phase(self, user_message: str) -> Dict:
        """
        Maneja la fase de recopilación de intereses
        """
        
        if len(self.conversation_state['preferences']['interests']) >= 2:
            self.conversation_state['phase'] = 'details'
        
        
        prompt = self.conversation_templates['interests'].format(
            destination=self.conversation_state['preferences']['destination'],
            conversation_history=self._format_conversation_history(),
            user_message=user_message
        )
        
        response = self.mistral_client.generate(prompt)
        guide_response = response.strip()
        
        
        self.conversation_state['conversation_history'].append({
            'role': 'assistant',
            'content': guide_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'type': 'guide_response',
            'message': guide_response,
            'phase': self.conversation_state['phase'],
            'preferences_collected': False,
            'current_preferences': self.conversation_state['preferences']
        }
    
    def _handle_details_phase(self, user_message: str) -> Dict:
        """
        Maneja la fase de recopilación de detalles
        """
        
        if self._has_minimum_preferences():
            self.conversation_state['phase'] = 'summary'
        
        
        prompt = self.conversation_templates['details'].format(
            destination=self.conversation_state['preferences']['destination'],
            interests=', '.join(self.conversation_state['preferences']['interests']),
            conversation_history=self._format_conversation_history(),
            user_message=user_message
        )
        
        response = self.mistral_client.generate(prompt)
        guide_response = response.strip()
        
        
        self.conversation_state['conversation_history'].append({
            'role': 'assistant',
            'content': guide_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'type': 'guide_response',
            'message': guide_response,
            'phase': self.conversation_state['phase'],
            'preferences_collected': self.conversation_state['phase'] == 'summary',
            'current_preferences': self.conversation_state['preferences']
        }
    
    def _handle_summary_phase(self, user_message: str) -> Dict:
        """
        Maneja la fase de resumen
        """
        
        prompt = self.conversation_templates['summary'].format(
            preferences=json.dumps(self.conversation_state['preferences'], indent=2, ensure_ascii=False)
        )
        
        response = self.mistral_client.generate(prompt)
        guide_response = response.strip()
        
        
        self.conversation_state['conversation_history'].append({
            'role': 'assistant',
            'content': guide_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'type': 'guide_response',
            'message': guide_response,
            'phase': 'complete',
            'preferences_collected': True,
            'final_preferences': self.conversation_state['preferences'],
            'current_preferences': self.conversation_state['preferences']  
        }
    
    def _handle_general_conversation(self, user_message: str) -> Dict:
        """
        Maneja conversación general fuera de las fases definidas
        """
        prompt = f"""
        Eres un guía turístico experto. El usuario ha dicho: "{user_message}"
        
        Contexto de preferencias actuales:
        {json.dumps(self.conversation_state['preferences'], indent=2, ensure_ascii=False)}
        
        Responde de manera útil y amigable, ayudando a completar la información de viaje si es necesario.
        Mantén la respuesta en 3-4 líneas máximo.
        """
        
        response = self.mistral_client.generate(prompt)
        guide_response = response.strip()
        
        
        self.conversation_state['conversation_history'].append({
            'role': 'assistant',
            'content': guide_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'type': 'guide_response',
            'message': guide_response,
            'phase': self.conversation_state['phase'],
            'preferences_collected': self._is_conversation_complete(),
            'current_preferences': self.conversation_state['preferences']
        }
    
    def _format_conversation_history(self) -> str:
        """
        Formatea el historial de conversación para los prompts
        """
        history = []
        
        recent_history = self.conversation_state['conversation_history'][-6:]
        
        for entry in recent_history:
            role = "Usuario" if entry['role'] == 'user' else "Guía"
            history.append(f"{role}: {entry['content']}")
        
        return "\n".join(history)
    
    def _has_minimum_preferences(self) -> bool:
        """
        Verifica si tenemos las preferencias mínimas necesarias
        """
        prefs = self.conversation_state['preferences']
        
        
        if hasattr(self, '_user_wants_to_proceed'):
            
            return (
                prefs['destination'] is not None or
                len(prefs['interests']) >= 1
            )
        
        
        return (
            prefs['destination'] is not None and
            len(prefs['interests']) >= 2
        )
    
    def _is_conversation_complete(self) -> bool:
        """
        Verifica si la conversación está completa
        """
        return (
            self._has_minimum_preferences() and
            self.conversation_state['phase'] in ['summary', 'complete']
        )
    
    def _wants_to_proceed_with_current_info(self, user_message: str) -> bool:
        """
        Detecta si el usuario quiere proceder con la información actual sin proporcionar más detalles
        """
        
        proceed_phrases = [
            
            'eso es todo',
            'es todo',
            'con eso es suficiente',
            'con eso basta',
            'no necesito más',
            'no quiero dar más información',
            'no quiero más detalles',
            'procede',
            'continúa',
            'adelante',
            'ya está',
            'listo',
            'es suficiente',
            'genera el itinerario',
            'hazme el itinerario',
            'crea el itinerario',
            'busca información',
            'empieza a buscar',
            'no tengo más información',
            'no sé más',
            'con lo que te dije',
            'con lo que tienes',
            'ya te dije todo',
            'no importa',
            'da igual',
            'lo que sea',
            'sorpréndeme',
            'tú decide',
            'no me preguntes más',
            'deja de preguntar',
            'muchas preguntas',
            'ya basta de preguntas',
            
            'quiero el itinerario',
            'dame el itinerario',
            'muéstrame el itinerario',
            'enséñame el itinerario',
            'ya no más',
            'no más preguntas',
            'suficiente información',
            'ya tienes todo',
            'con eso alcanza',
            'no tengo más detalles',
            'no sé qué más decir',
            'ya dije todo',
            'nada más',
            'eso nomás',
            'solo eso',
            'ya pues',
            'dale',
            'ok ya',
            'bueno ya',
            'ya ya',
            'está bien así',
            'así está bien',
            'déjalo así',
            'así nomás',
            'no hay más',
            'termina ya',
            'finaliza',
            'acaba',
            'ya termina',
            'apúrate',
            'rápido',
            'de una vez',
            'ya por favor',
            'por favor ya',
            'hazlo ya',
            'solo hazlo',
            'empieza ya',
            'comienza ya',
            'inicia ya',
            'busca ya',
            'encuentra ya',
            'muestra ya'
        ]
        
        message_lower = user_message.lower().strip()
        
        
        for phrase in proceed_phrases:
            if phrase in message_lower:
                return True
        
        
        impatience_patterns = [
            r'\b(ya|basta|suficiente|no más|no mas)\b',
            r'\b(dale|anda|vamos|venga)\b',
            r'\b(itinerario|plan|viaje)\s*(ya|ahora|por favor)',
            r'^(si|sí|ok|okay|bueno|vale|bien)$',
            r'^(no|nada|ninguno|ninguna)$',
            r'\b(no\s*(sé|se)|no\s*tengo)\b.*\b(más|mas|idea)\b'
        ]
        
        import re
        for pattern in impatience_patterns:
            if re.search(pattern, message_lower):
                
                if len(message_lower.split()) <= 5:  
                    return True
        
        
        words = message_lower.split()
        if len(words) <= 3:
            impatience_words = ['ya', 'no', 'basta', 'suficiente', 'listo', 'dale', 'ok', 'bueno', 'si', 'sí']
            if any(word in impatience_words for word in words):
                return True
        
        
        evasive_patterns = [
            r'no\s*(importa|interesa)',
            r'(cualquier|cualquiera)\s*(cosa|lugar|sitio)',
            r'donde\s*(sea|quieras)',
            r'lo\s*que\s*(sea|quieras|recomiendes)',
            r'(tú|tu)\s*(decide|eliges|escoges|recomienda)',
            r'no\s*tengo\s*(preferencia|idea)',
            r'me\s*da\s*(igual|lo mismo)'
        ]
        
        for pattern in evasive_patterns:
            if re.search(pattern, message_lower):
                return True
        
        
        try:
            prompt = f"""
            Analiza el siguiente mensaje del usuario en el contexto de una conversación sobre planificación de viajes.
            
            Mensaje del usuario: "{user_message}"
            
            Determina si el usuario:
            1. Quiere proceder con la planificación sin dar más información
            2. Muestra impaciencia o frustración con las preguntas
            3. Indica que ya proporcionó suficiente información
            4. Quiere que se genere el itinerario con lo que ya se tiene
            5. Está siendo evasivo o no quiere compartir más detalles
            
            Considera también:
            - Mensajes muy cortos pueden indicar impaciencia
            - Respuestas como "ya", "no", "listo" suelen indicar que quiere proceder
            - Frases que piden acción inmediata ("hazlo ya", "busca ahora")
            
            Responde ÚNICAMENTE con 'true' si detectas cualquiera de estas intenciones,
            o 'false' si el usuario parece dispuesto a continuar proporcionando información.
            """
            
            response = self.mistral_client.generate(prompt)
            result = response.strip().lower()
            
            return 'true' in result
            
        except Exception:
            
            return False
    
    def _explain_missing_info(self) -> Dict:
        """
        Explica qué información mínima falta para poder generar el itinerario
        """
        prefs = self.conversation_state['preferences']
        
        
        if not prefs['destination'] and len(prefs['interests']) == 0:
            message = "Entiendo que quieres proceder rápidamente. Solo necesito saber a dónde quieres viajar o qué tipo de actividades te interesan para poder ayudarte. ¿Puedes darme al menos uno de estos datos?"
        
        
        elif not prefs['destination'] and len(prefs['interests']) > 0:
            
            interests_text = ", ".join(prefs['interests'])
            message = f"Perfecto, veo que te interesan: {interests_text}. Buscaré las mejores opciones para estas actividades. ¡Empecemos con tu itinerario!"
            self.conversation_state['phase'] = 'summary'
            return self._handle_summary_phase("")
        
        elif prefs['destination'] and len(prefs['interests']) == 0:
            
            message = f"Entendido, prepararé un itinerario general para {prefs['destination']} con las atracciones más populares. ¡Vamos allá!"
            
            prefs['interests'] = ['restaurants', 'activities', 'museums']
            self.conversation_state['phase'] = 'summary'
            return self._handle_summary_phase("")
        
        else:
            
            self.conversation_state['phase'] = 'summary'
            return self._handle_summary_phase("")
        
        
        self.conversation_state['conversation_history'].append({
            'role': 'assistant',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'type': 'guide_response',
            'message': message,
            'phase': self.conversation_state['phase'],
            'preferences_collected': False,
            'current_preferences': self.conversation_state['preferences']
        }
    
    def _reset_conversation(self) -> Dict:
        """
        Reinicia la conversación
        """
        self.conversation_state = {
            'phase': 'greeting',
            'preferences': {
                'destination': None,
                'interests': [],
                'accommodation_type': None,
                'budget': None,
                'duration': None,
                'travel_dates': None,
                'special_requirements': [],
                'preferred_activities': []
            },
            'conversation_history': [],
            'questions_asked': set(),
            'extraction_attempts': 0
        }
        
        return {
            'type': 'reset_complete',
            'message': 'Conversación reiniciada'
        }
    
    def get_structured_preferences(self) -> Dict:
        """
        Devuelve las preferencias en un formato estructurado para el crawler
        """
        prefs = self.conversation_state['preferences']
        
        
        keywords = []
        
        
        if prefs['destination']:
            keywords.append(prefs['destination'])
        
        
        keywords.extend(prefs['interests'])
        
        
        keywords.extend(prefs['preferred_activities'])
        
        
        query_parts = []
        if prefs['destination']:
            query_parts.append(f"turismo en {prefs['destination']}")
        
        for interest in prefs['interests']:
            destination = prefs['destination'] or 'el destino'
            if interest == 'accommodation':
                query_parts.append(f"hoteles en {destination}")
            elif interest == 'beaches':
                query_parts.append(f"mejores playas de {destination}")
            elif interest == 'museums':
                query_parts.append(f"museos en {destination}")
            elif interest == 'restaurants':
                query_parts.append(f"restaurantes recomendados en {destination}")
            else:
                query_parts.append(f"{interest} en {destination}")
        
        improved_query = " ".join(query_parts)
        
        return {
            'keywords': list(set(keywords)),  
            'improved_query': improved_query,
            'raw_preferences': prefs,
            'search_context': {
                'destination': prefs['destination'],
                'main_interests': prefs['interests'][:3],  
                'budget': prefs['budget'],
                'duration': prefs['duration']
            }
        }