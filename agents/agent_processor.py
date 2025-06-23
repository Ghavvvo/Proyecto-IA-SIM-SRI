"""
Agente procesador que utiliza Gemini para estructurar la información turística
antes de guardarla en ChromaDB
"""
from core.gemini_config import GeminiClient, gemini_json
import json
from typing import Dict, List, Optional
from autogen import Agent
import re


class ProcessorAgent(Agent):
    """
    Agente que procesa el contenido extraído por el crawler usando Gemini
    para estructurar la información turística en formato JSON
    """
    
    def __init__(self, name: str = "ProcessorAgent"):
        super().__init__(name)
        self.gemini_client = GeminiClient(model_name="flash")
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
            
            # Generar respuesta JSON con Gemini
            structured_data = self.gemini_client.generate_json(prompt)
            
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
        La estructura debe seguir este formato EXACTO, pero SOLO incluye los campos para los que encuentres información:
        
        {{
            "pais": "nombre del país (si se menciona)",
            "ciudad": "nombre de la ciudad principal (si se menciona)",
            "lugares": [
                {{
                    "nombre": "nombre del lugar",
                    "tipo": "hotel/restaurante/playa/museo/monumento/parque/atraccion/actividad/tour",
                    "subtipo": "tipo más específico si aplica (ej: hotel boutique, restaurante italiano, playa privada)",
                    "ubicacion": {{
                        "direccion": "dirección completa si está disponible",
                        "zona": "barrio o zona de la ciudad",
                        "ciudad": "ciudad donde está ubicado",
                        "coordenadas": "coordenadas GPS si están disponibles"
                    }},
                    "descripcion": "descripción breve del lugar",
                    "caracteristicas": ["lista", "de", "características", "principales"],
                    "precios": {{
                        "moneda": "USD/EUR/etc",
                        "rango_precio": "económico/moderado/costoso/lujo",
                        "precio_desde": "precio mínimo si se menciona",
                        "precio_hasta": "precio máximo si se menciona",
                        "precio_promedio": "precio promedio si se menciona",
                        "detalles_precio": "información adicional sobre precios"
                    }},
                    "horarios": {{
                        "apertura": "hora de apertura",
                        "cierre": "hora de cierre",
                        "dias_operacion": ["lista de días que opera"],
                        "temporada": "si es estacional, indicar temporada"
                    }},
                    "contacto": {{
                        "telefono": "número de teléfono",
                        "email": "correo electrónico",
                        "website": "sitio web",
                        "redes_sociales": {{"facebook": "", "instagram": "", "twitter": ""}}
                    }},
                    "calificacion": {{
                        "puntuacion": "puntuación numérica si existe",
                        "escala": "escala de la puntuación (ej: de 5)",
                        "num_resenas": "número de reseñas si se menciona"
                    }},
                    "servicios": ["lista", "de", "servicios", "que", "ofrece"],
                    "idiomas": ["idiomas", "que", "se", "hablan"],
                    "accesibilidad": "información sobre accesibilidad para personas con discapacidad",
                    "recomendaciones": "recomendaciones o tips especiales",
                    "mejor_epoca_visita": "mejor época para visitar si se menciona"
                }}
            ],
            "informacion_general": {{
                "clima": "información sobre el clima",
                "moneda_local": "moneda que se usa en el lugar",
                "idioma_principal": "idioma principal del lugar",
                "mejor_epoca_visita": "mejor época para visitar en general",
                "duracion_recomendada": "cuántos días se recomienda para visitar",
                "presupuesto_diario": "presupuesto diario estimado",
                "tips_viajeros": ["lista", "de", "consejos", "para", "viajeros"],
                "como_llegar": "información sobre cómo llegar al destino",
                "transporte_local": "información sobre transporte en el destino",
                "seguridad": "información sobre seguridad para turistas"
            }},
            "actividades_populares": [
                {{
                    "nombre": "nombre de la actividad",
                    "tipo": "tipo de actividad",
                    "duracion": "duración estimada",
                    "precio": "precio si se menciona",
                    "descripcion": "descripción de la actividad",
                    "incluye": ["que", "incluye", "la", "actividad"],
                    "requerimientos": "requerimientos o restricciones"
                }}
            ],
            "gastronomia": {{
                "platos_tipicos": ["lista", "de", "platos", "típicos"],
                "bebidas_tipicas": ["lista", "de", "bebidas", "típicas"],
                "especialidades_locales": "especialidades gastronómicas del lugar"
            }},
            "eventos_festivales": [
                {{
                    "nombre": "nombre del evento o festival",
                    "fecha": "fecha o época del año",
                    "descripcion": "descripción del evento",
                    "duracion": "duración del evento"
                }}
            ],
            "compras": {{
                "mercados_locales": ["lista de mercados"],
                "centros_comerciales": ["lista de centros comerciales"],
                "souvenirs_tipicos": ["lista de souvenirs típicos"],
                "zonas_comerciales": ["zonas recomendadas para compras"]
            }},
            "alojamiento_resumen": {{
                "tipos_disponibles": ["hotel", "hostal", "airbnb", "resort", "etc"],
                "rango_precios": "rango general de precios de alojamiento",
                "zonas_recomendadas": ["zonas recomendadas para hospedarse"],
                "temporada_alta": "cuándo es temporada alta",
                "temporada_baja": "cuándo es temporada baja"
            }}
        }}
        
        IMPORTANTE:
        - Solo incluye información que esté EXPLÍCITAMENTE mencionada en el texto
        - Si no encuentras información sobre un campo, NO lo incluyas en el JSON
        - Si no encuentras información clara sobre país o ciudad, intenta inferirlo del contexto o URL
        - Los precios deben incluir la moneda si se menciona
        - Para el campo "tipo" usa EXACTAMENTE uno de estos valores: hotel, restaurante, playa, museo, monumento, parque, atraccion, actividad, tour, tienda, mercado, centro_comercial, bar, discoteca, cafe, transporte
        - Asegúrate de que el JSON sea válido y esté bien formateado
        - Si el contenido no es sobre turismo o no tiene información relevante, devuelve: {{"tipo_contenido": "no_turistico"}}
        
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