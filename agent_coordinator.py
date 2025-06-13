from autogen import Agent
from typing import List
import google.generativeai as genai

class CoordinatorAgent(Agent):
    def __init__(self, name, crawler_agent, rag_agent, interface_agent):
        super().__init__(name)
        self.crawler_agent = crawler_agent
        self.rag_agent = rag_agent
        self.interface_agent = interface_agent

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
        # Consultar al agente RAG
        self._notify_interface('query_received', {
            'query': query,
            'status': 'processing'
        })
        response = self.rag_agent.receive({'type': 'query', 'query': query}, self)

        if response['type'] == 'answer':
            # Utilizar Gemini para evaluar si la respuesta es útil
            evaluation = self._evaluate_response_usefulness(query, response['answer'])
            if not evaluation:
                # Extraer palabras clave problemáticas cuando la consulta no es relevante
                problematic_keywords = self._extract_problematic_keywords(query, response['answer'])

                print(f"La respuesta proporcionada por el sistema RAG no es útil para la consulta.")
                print(f"Palabras clave problemáticas identificadas: {', '.join(problematic_keywords)}")
                print("Actualizando la base de datos con información relevante...")

                # Informar al agente crawler para buscar en Google y actualizar la base de datos
                crawl_result = self.crawler_agent.receive({'type': 'crawl_keywords', 'keywords': problematic_keywords}, self)

                if crawl_result.get('type') == 'crawled':
                    # La base de datos se actualizó correctamente
                    pages_processed = crawl_result.get('pages_processed', 0)
                    print(f"Base de datos actualizada exitosamente con {pages_processed} nuevas páginas.")

                    # Volver a consultar al agente RAG con la misma consulta
                    print("Consultando nuevamente con la información actualizada...")
                    new_response = self.rag_agent.receive({'type': 'query', 'query': query}, self)

                    if new_response['type'] == 'answer':
                        return new_response['answer']
                    else:
                        return new_response.get('msg', 'Error en la nueva consulta después de actualizar la base de datos')
                else:
                    error_msg = crawl_result.get('msg', 'Error desconocido')
                    print(f"No se pudo actualizar la base de datos: {error_msg}")
                    # Devolver la respuesta original si no se pudo actualizar
                    return response['answer']

            return response['answer']

        return response.get('msg', 'Error al consultar la base de datos')

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
        
    def _notify_interface(self, event_type, event_data):
        """Envía notificaciones sin system_state"""

        message = {
            'event_type': event_type,
            'event_data': event_data or {}
        }
        return self.interface_agent.receive(message, self)
