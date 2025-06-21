"""
Agente procesador que utiliza Gemini para estructurar la información turística
antes de guardarla en ChromaDB
"""
import google.generativeai as genai
import json
from typing import Dict, List, Optional
from autogen import Agent
import re

# Configuración de Gemini
GEMINI_API_KEY = "AIzaSyCuiFY0aCJEaOndmd_jEHZIabbA23TWn6E"
genai.configure(api_key=GEMINI_API_KEY)


class ProcessorAgent(Agent):
    """
    Agente que procesa el contenido extraído por el crawler usando Gemini
    para estructurar la información turística en formato JSON
    """
    
    def __init__(self, name: str = "ProcessorAgent"):
        super().__init__(name)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.processed_count = 0
        self.errors_count = 0
        
    def receive(self, message, sender):
        """Recibe y procesa mensajes del crawler"""
        if message['type'] == 'process_content':
            # Procesar contenido individual
            content_data = message.get('content_data')
            if not content_data:
                return {'type': 'error', 'msg': 'No se proporcionó contenido para procesar'}
            
            processed_data = self._process_single_content(content_data)
            return {
                'type': 'processed',
                'data': processed_data,
                'success': processed_data is not None
            }
            
        elif message['type'] == 'process_batch':
            # Procesar múltiples contenidos
            contents = message.get('contents', [])
            processed_results = []
            
            print(f"🤖 Procesando {len(contents)} páginas con Gemini...")
            
            for content_data in contents:
                result = self._process_single_content(content_data)
                if result:
                    processed_results.append(result)
            
            return {
                'type': 'batch_processed',
                'results': processed_results,
                'total_processed': self.processed_count,
                'errors': self.errors_count
            }
            
        return {'type': 'error', 'msg': 'Tipo de mensaje desconocido'}
    
    def _process_single_content(self, content_data: Dict) -> Optional[Dict]:
        """
        Procesa un contenido individual con Gemini para extraer información turística estructurada
        """
        try:
            url = content_data.get('url', '')
            title = content_data.get('title', '')
            content = content_data.get('content', '')
            
            if not content or len(content) < 100:
                return None
            
            # Crear prompt para Gemini
            prompt = self._create_extraction_prompt(content, title, url)
            
            # Generar respuesta con Gemini
            response = self.model.generate_content(prompt)
            
            # Parsear la respuesta JSON
            structured_data = self._parse_gemini_response(response.text)
            
            if structured_data:
                # Añadir metadatos
                structured_data['source_url'] = url
                structured_data['source_title'] = title
                structured_data['processed_by'] = 'gemini'
                
                self.processed_count += 1
                print(f"✅ Procesado: {title[:50]}...")
                
                return structured_data
            else:
                self.errors_count += 1
                return None
                
        except Exception as e:
            self.errors_count += 1
            print(f"❌ Error procesando contenido: {str(e)}")
            return None
    
    def _create_extraction_prompt(self, content: str, title: str, url: str) -> str:
        """
        Crea el prompt para Gemini para extraer información turística estructurada
        """
        # Limitar el contenido para no exceder límites de tokens
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = f"""
        Analiza el siguiente contenido de una página web sobre turismo y extrae información estructurada.
        
        Título: {title}
        URL: {url}
        
        Contenido:
        {content}
        
        Extrae ÚNICAMENTE la información que esté presente en el texto anterior y devuélvela en formato JSON.
        La estructura debe seguir este formato, pero SOLO incluye los campos para los que encuentres información:
        
        {{
            "paises": [
                {{
                    "nombre": "nombre del país",
                    "hoteles": [
                        {{
                            "nombre": "nombre del hotel",
                            "localidad": "ciudad o zona",
                            "clasificacion": "estrellas o categoría",
                            "precio_promedio": "precio por noche si está disponible"
                        }}
                    ],
                    "lugares_turisticos": [
                        {{
                            "nombre": "nombre del lugar",
                            "localidad": "ciudad o zona",
                            "tipo": "playa/museo/monumento/parque/etc",
                            "precio_entrada": "precio si está disponible"
                        }}
                    ],
                    "precio_promedio_visita": "costo promedio de visitar el país si se menciona",
                    "mejor_epoca": "mejor época para visitar si se menciona",
                    "informacion_adicional": "otra información relevante"
                }}
            ]
        }}
        
        IMPORTANTE:
        - Solo incluye información que esté EXPLÍCITAMENTE mencionada en el texto
        - Si no encuentras información sobre un campo, NO lo incluyas en el JSON
        - Si no encuentras información sobre ningún país, devuelve un JSON vacío: {{}}
        - Los precios deben incluir la moneda si se menciona
        - Asegúrate de que el JSON sea válido y esté bien formateado
        
        Responde SOLO con el JSON, sin explicaciones adicionales.
        """
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """
        Parsea la respuesta de Gemini y extrae el JSON estructurado
        """
        try:
            # Limpiar la respuesta para extraer solo el JSON
            # Buscar el JSON entre llaves
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Parsear el JSON
                data = json.loads(json_str)
                
                # Validar que tenga la estructura esperada
                if isinstance(data, dict):
                    # Si no tiene países pero tiene otras claves, crear estructura de países
                    if 'paises' not in data and len(data) > 0:
                        # Intentar inferir el país del contenido
                        data = {'paises': [data]}
                    
                    return data
                    
            return None
            
        except json.JSONDecodeError as e:
            print(f"Error parseando JSON de Gemini: {e}")
            # Intentar corregir errores comunes
            try:
                # Eliminar comas finales
                cleaned = re.sub(r',\s*}', '}', response_text)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            except:
                pass
                
            return None
        except Exception as e:
            print(f"Error procesando respuesta de Gemini: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del procesamiento"""
        return {
            'processed_count': self.processed_count,
            'errors_count': self.errors_count,
            'success_rate': self.processed_count / (self.processed_count + self.errors_count) 
                           if (self.processed_count + self.errors_count) > 0 else 0
        }