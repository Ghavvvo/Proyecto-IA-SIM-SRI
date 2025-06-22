from autogen import Agent
import textwrap
from datetime import datetime

from mistral_utils import GenerativeModel


class InterfaceAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.model = GenerativeModel("mistral-large-latest")
        self.conversation_context = []
    
    def receive(self, message, sender):
        """
        Procesa cualquier notificación del sistema y genera una respuesta natural.
        
        Args:
            message: Dict con {
                'event_type': tipo de evento (str),
                'event_data': datos relevantes (dict)
            }
            sender: Agente que envía el mensaje
        """
        # Actualizar contexto de conversación
        self._update_context(message, sender)
        
        # Generar respuesta natural
        response = self._generate_natural_response(message)
        print(f"\n[Asistente]: {response}\n")
        return response
    
    def _update_context(self, message, sender):
        """Mantiene un contexto de lo que está ocurriendo en el sistema"""
        context_entry = {
            'event': message.get('event_type'),
            'sender': sender.name,
            'data': message.get('event_data', {}),
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_context.append(context_entry)
        
        # Mantener solo los últimos 5 eventos para contexto
        if len(self.conversation_context) > 5:
            self.conversation_context.pop(0)
    
    def _generate_natural_response(self, message):
        """Genera una respuesta completamente dinámica usando el modelo generativo"""
        try:
            prompt = self._build_dynamic_prompt(message)
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generando mensaje: {e}")
            return "Estamos procesando su solicitud. Un momento, por favor."
    
    def _build_dynamic_prompt(self, message):
        """Construye un prompt contextualizado para el modelo generativo"""
        event_type = message.get('event_type', 'unknown_event')
        event_data = message.get('event_data', {})
        
        return textwrap.dedent(f"""
        Eres un guía turístico profesional que ayuda a visitantes. El sistema de inteligencia artificial 
        detrás de ti acaba de realizar una acción y necesitas informar al usuario de manera natural.
        
        Contexto completo del sistema (últimas acciones):
        {self._format_context_for_prompt()}
        
        Último evento ocurrido:
        - Tipo: {event_type}
        - Datos: {event_data}
        
        Instrucciones:
        - Genera UN SÓLO MENSAJE BREVE (1-2 oraciones) en español
        - Usa un tono amable, profesional y entusiasta
        - Adapta el mensaje al tipo de evento ocurrido
        - Si es relevante, incluye detalles específicos de los datos del evento
        - Mantén la naturalidad de una conversación humana
        
        Ejemplos de estilo:
        - "Estoy consultando las últimas actualizaciones sobre horarios de visita..."
        - "Acabo de encontrar información excelente sobre restaurantes locales..."
        - "Voy a verificar esa información poco común que me mencionas..."
        
        """)
    
    def _format_context_for_prompt(self):
        """Formatea el contexto para incluirlo en el prompt"""
        return "\n".join(
            f"{item['timestamp']} - {item['sender']}: {item['event']} ({item['data']})"
            for item in self.conversation_context
        )