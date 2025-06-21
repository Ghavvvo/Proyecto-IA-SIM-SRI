"""
Agente procesador que utiliza GLiNER para extraer entidades de la información turística
antes de guardarla en ChromaDB
"""
from typing import Dict, List, Optional, Any
from autogen import Agent
import json
import re
from gliner import GLiNER


class GLiNERAgent(Agent):
    """
    Agente que procesa el contenido extraído por el crawler usando GLiNER
    para extraer entidades relevantes de turismo
    """
    
    def __init__(self, name: str = "GLiNERAgent", model_name: str = "urchade/gliner_multi-v2.1"):
        super().__init__(name)
        
        # Inicializar el modelo GLiNER
        print(f"🔄 Cargando modelo GLiNER: {model_name}...")
        self.model = GLiNER.from_pretrained(model_name)
        print("✅ Modelo GLiNER cargado exitosamente")
        
        # Definir las etiquetas de entidades para turismo
        self.tourism_labels = [
            # Lugares y destinos
            "PAÍS", "CIUDAD", "REGIÓN", "DESTINO_TURÍSTICO",
            "PLAYA", "MONTAÑA", "ISLA", "PUEBLO",
            
            # Alojamiento
            "HOTEL", "RESORT", "HOSTAL", "ALOJAMIENTO",
            "CADENA_HOTELERA",
            
            # Atracciones y lugares de interés
            "ATRACCIÓN_TURÍSTICA", "MUSEO", "MONUMENTO",
            "PARQUE", "SITIO_HISTÓRICO", "PATRIMONIO",
            "CENTRO_COMERCIAL", "MERCADO",
            
            # Servicios y actividades
            "RESTAURANTE", "BAR", "CAFÉ",
            "TOUR", "EXCURSIÓN", "ACTIVIDAD",
            "EVENTO", "FESTIVAL",
            
            # Transporte
            "AEROPUERTO", "ESTACIÓN", "PUERTO",
            "LÍNEA_AÉREA", "CRUCERO",
            
            # Información práctica
            "PRECIO", "MONEDA", "HORARIO",
            "TEMPORADA", "CLIMA",
            
            # Organizaciones
            "AGENCIA_VIAJES", "OPERADOR_TURÍSTICO",
            "OFICINA_TURISMO"
        ]
        
        # Etiquetas en inglés para mejor compatibilidad
        self.english_labels = [
            # Places and destinations
            "COUNTRY", "CITY", "REGION", "TOURIST_DESTINATION",
            "BEACH", "MOUNTAIN", "ISLAND", "TOWN",
            
            # Accommodation
            "HOTEL", "RESORT", "HOSTEL", "ACCOMMODATION",
            "HOTEL_CHAIN",
            
            # Attractions and points of interest
            "TOURIST_ATTRACTION", "MUSEUM", "MONUMENT",
            "PARK", "HISTORICAL_SITE", "HERITAGE_SITE",
            "SHOPPING_CENTER", "MARKET",
            
            # Services and activities
            "RESTAURANT", "BAR", "CAFE",
            "TOUR", "EXCURSION", "ACTIVITY",
            "EVENT", "FESTIVAL",
            
            # Transportation
            "AIRPORT", "STATION", "PORT",
            "AIRLINE", "CRUISE",
            
            # Practical information
            "PRICE", "CURRENCY", "SCHEDULE",
            "SEASON", "WEATHER",
            
            # Organizations
            "TRAVEL_AGENCY", "TOUR_OPERATOR",
            "TOURISM_OFFICE"
        ]
        
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
            
            print(f"🤖 Procesando {len(contents)} páginas con GLiNER...")
            
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
        Procesa un contenido individual con GLiNER para extraer entidades turísticas
        """
        try:
            url = content_data.get('url', '')
            title = content_data.get('title', '')
            content = content_data.get('content', '')
            
            if not content or len(content) < 50:
                return None
            
            # Combinar título y contenido para mejor extracción
            full_text = f"{title}\n\n{content}"
            
            # Limitar el texto para evitar problemas de memoria
            max_length = 2000
            if len(full_text) > max_length:
                full_text = full_text[:max_length]
            
            # Extraer entidades con GLiNER
            entities_spanish = self._extract_entities(full_text, self.tourism_labels)
            entities_english = self._extract_entities(full_text, self.english_labels)
            
            # Combinar resultados
            all_entities = self._merge_entities(entities_spanish, entities_english)
            
            # Estructurar la información extraída
            structured_data = self._structure_entities(all_entities, content_data)
            
            if structured_data:
                self.processed_count += 1
                print(f"✅ Procesado con GLiNER: {title[:50]}...")
                return structured_data
            else:
                self.errors_count += 1
                return None
                
        except Exception as e:
            self.errors_count += 1
            print(f"❌ Error procesando contenido con GLiNER: {str(e)}")
            return None
    
    def _extract_entities(self, text: str, labels: List[str]) -> List[Dict]:
        """
        Extrae entidades del texto usando GLiNER
        """
        try:
            # Configurar umbral de confianza
            threshold = 0.5
            
            # Predecir entidades
            entities = self.model.predict_entities(
                text, 
                labels, 
                threshold=threshold
            )
            
            return entities
            
        except Exception as e:
            print(f"Error en extracción de entidades: {e}")
            return []
    
    def _merge_entities(self, entities_spanish: List[Dict], entities_english: List[Dict]) -> List[Dict]:
        """
        Combina las entidades extraídas en español e inglés, eliminando duplicados
        """
        all_entities = []
        seen_texts = set()
        
        # Mapeo de etiquetas español-inglés
        label_mapping = {
            "PAÍS": "COUNTRY",
            "CIUDAD": "CITY",
            "REGIÓN": "REGION",
            "DESTINO_TURÍSTICO": "TOURIST_DESTINATION",
            "PLAYA": "BEACH",
            "MONTAÑA": "MOUNTAIN",
            "ISLA": "ISLAND",
            "PUEBLO": "TOWN",
            "ATRACCIÓN_TURÍSTICA": "TOURIST_ATTRACTION",
            "MUSEO": "MUSEUM",
            "MONUMENTO": "MONUMENT",
            "PARQUE": "PARK",
            "SITIO_HISTÓRICO": "HISTORICAL_SITE",
            "PATRIMONIO": "HERITAGE_SITE",
            "CENTRO_COMERCIAL": "SHOPPING_CENTER",
            "MERCADO": "MARKET",
            "RESTAURANTE": "RESTAURANT",
            "PRECIO": "PRICE",
            "MONEDA": "CURRENCY",
            "HORARIO": "SCHEDULE",
            "TEMPORADA": "SEASON",
            "CLIMA": "WEATHER",
            "AGENCIA_VIAJES": "TRAVEL_AGENCY",
            "OPERADOR_TURÍSTICO": "TOUR_OPERATOR",
            "OFICINA_TURISMO": "TOURISM_OFFICE"
        }
        
        # Procesar entidades en español
        for entity in entities_spanish:
            text = entity.get('text', '').strip()
            if text and text.lower() not in seen_texts:
                seen_texts.add(text.lower())
                # Normalizar etiqueta a inglés
                label = entity.get('label', '')
                normalized_label = label_mapping.get(label, label)
                all_entities.append({
                    'text': text,
                    'label': normalized_label,
                    'score': entity.get('score', 0)
                })
        
        # Procesar entidades en inglés
        for entity in entities_english:
            text = entity.get('text', '').strip()
            if text and text.lower() not in seen_texts:
                seen_texts.add(text.lower())
                all_entities.append(entity)
        
        return all_entities
    
    def _structure_entities(self, entities: List[Dict], content_data: Dict) -> Dict:
        """
        Estructura las entidades extraídas en un formato organizado para la base de datos
        """
        structured = {
            'source_url': content_data.get('url', ''),
            'source_title': content_data.get('title', ''),
            'processed_by': 'gliner',
            'entities': {
                'countries': [],
                'cities': [],
                'destinations': [],
                'hotels': [],
                'attractions': [],
                'restaurants': [],
                'prices': [],
                'activities': [],
                'transport': [],
                'organizations': []
            },
            'raw_entities': []
        }
        
        # Categorizar entidades
        for entity in entities:
            text = entity.get('text', '')
            label = entity.get('label', '')
            score = entity.get('score', 0)
            
            # Añadir a entidades crudas
            structured['raw_entities'].append({
                'text': text,
                'type': label,
                'confidence': score
            })
            
            # Categorizar por tipo
            if label == 'COUNTRY':
                structured['entities']['countries'].append(text)
            elif label == 'CITY':
                structured['entities']['cities'].append(text)
            elif label in ['TOURIST_DESTINATION', 'BEACH', 'MOUNTAIN', 'ISLAND']:
                structured['entities']['destinations'].append({
                    'name': text,
                    'type': label.lower()
                })
            elif label in ['HOTEL', 'RESORT', 'HOSTEL', 'ACCOMMODATION']:
                structured['entities']['hotels'].append({
                    'name': text,
                    'type': label.lower()
                })
            elif label in ['TOURIST_ATTRACTION', 'MUSEUM', 'MONUMENT', 'PARK', 
                          'HISTORICAL_SITE', 'HERITAGE_SITE']:
                structured['entities']['attractions'].append({
                    'name': text,
                    'type': label.lower()
                })
            elif label in ['RESTAURANT', 'BAR', 'CAFE']:
                structured['entities']['restaurants'].append({
                    'name': text,
                    'type': label.lower()
                })
            elif label in ['PRICE', 'CURRENCY']:
                structured['entities']['prices'].append({
                    'text': text,
                    'type': label.lower()
                })
            elif label in ['TOUR', 'EXCURSION', 'ACTIVITY', 'EVENT', 'FESTIVAL']:
                structured['entities']['activities'].append({
                    'name': text,
                    'type': label.lower()
                })
            elif label in ['AIRPORT', 'STATION', 'PORT', 'AIRLINE', 'CRUISE']:
                structured['entities']['transport'].append({
                    'name': text,
                    'type': label.lower()
                })
            elif label in ['TRAVEL_AGENCY', 'TOUR_OPERATOR', 'TOURISM_OFFICE']:
                structured['entities']['organizations'].append({
                    'name': text,
                    'type': label.lower()
                })
        
        # Eliminar listas vacías
        structured['entities'] = {k: v for k, v in structured['entities'].items() if v}
        
        # Generar resumen
        structured['summary'] = self._generate_summary(structured['entities'])
        
        return structured
    
    def _generate_summary(self, entities: Dict) -> str:
        """
        Genera un resumen textual de las entidades extraídas
        """
        summary_parts = []
        
        if 'countries' in entities:
            countries = ', '.join(entities['countries'][:3])
            summary_parts.append(f"Países: {countries}")
        
        if 'cities' in entities:
            cities = ', '.join(entities['cities'][:5])
            summary_parts.append(f"Ciudades: {cities}")
        
        if 'hotels' in entities:
            hotel_names = [h['name'] for h in entities['hotels'][:3]]
            hotels = ', '.join(hotel_names)
            summary_parts.append(f"Hoteles: {hotels}")
        
        if 'attractions' in entities:
            attraction_names = [a['name'] for a in entities['attractions'][:3]]
            attractions = ', '.join(attraction_names)
            summary_parts.append(f"Atracciones: {attractions}")
        
        return '. '.join(summary_parts) if summary_parts else "Sin entidades relevantes encontradas"
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del procesamiento"""
        return {
            'processed_count': self.processed_count,
            'errors_count': self.errors_count,
            'success_rate': self.processed_count / (self.processed_count + self.errors_count) 
                           if (self.processed_count + self.errors_count) > 0 else 0
        }