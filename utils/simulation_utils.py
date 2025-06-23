import re
from typing import Dict, Any
import google.generativeai as genai

def format_as_simulation_input(raw_response: str, preferences: dict) -> Dict[str, Any]:
    """
    Uses Gemini to transform a formatted itinerary into a structured format optimal for tourist simulation.
    Enriched with detailed information for simulation including types, popularity, distances, and timing.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Eres un experto en simulación de turistas. Recibe el siguiente itinerario de viaje y extrae información detallada para simular a un turista siguiendo el itinerario.

        Itinerario:
        {raw_response}

        Preferencias del viajero:
        {preferences}

        INSTRUCCIONES:
        - Devuelve un JSON estructurado con la siguiente forma:
          {{
            "destination": string,
            "season": string (verano/invierno/primavera/otoño, inferir de las fechas o usar "verano" por defecto),
            "tourist_profile": string (exigente/relajado/average basado en preferencias),
            "days": [
              {{
                "day": int (número del día, empezando desde 1),
                "title": string (titulo del dia),
                "day_of_week": string (lunes/martes/etc, usar "sabado" si no se especifica),
                "activities": [
                  {{
                    "time": string (formato HH:MM, si no hay hora usar "09:00", "11:00", "14:00", etc secuencialmente),
                    "location": string (nombre del lugar o actividad),
                    "description": string (descripcion breve del lugar o actividad),
                    "type": string (museo/restaurante/parque/monumento/playa/hotel/centro_comercial/teatro/zoo/otro),
                    "popularity": float (1-10, estimar basado en la descripción, lugares famosos 8-10, normales 5-7),
                    "estimated_duration_hours": float (museo:1.5, restaurante:1.2, parque:1.0, monumento:0.5, playa:2.0, etc),
                    "distance_from_previous_km": float (estimar distancia, primera actividad usar 3.0, entre actividades 1-5 km)
                  }}
                ]
              }}
            ],
            "interests": list (extraer de preferencias),
            "budget": string (si está disponible),
            "duration": string (duración total del viaje)
          }}
        
        REGLAS ADICIONALES:
        - Para el tipo de lugar, inferir del nombre y descripción (ej: "Museo Nacional" -> "museo")
        - Para popularidad: lugares muy conocidos o mencionados como "imperdibles" = 8-10
        - Para distancias: en la misma zona 0.5-2km, zonas diferentes 3-8km
        - Para tourist_profile: si menciona lujo/exclusivo = "exigente", si menciona relajado/tranquilo = "relajado", sino "average"
        - Si hay horarios específicos, úsalos. Si no, distribuye las actividades lógicamente durante el día
        - Devuelve SOLO el JSON, sin explicaciones ni comentarios.
        """
        response = model.generate_content(prompt)
        json_str = response.text.strip()
        # Buscar el JSON en la respuesta
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx]
            import json
            simulation_data = json.loads(json_str)
            
            # Enriquecer con valores por defecto si faltan
            if 'season' not in simulation_data:
                simulation_data['season'] = 'verano'
            if 'tourist_profile' not in simulation_data:
                simulation_data['tourist_profile'] = 'average'
                
            # Asegurar que cada actividad tenga todos los campos necesarios
            for day in simulation_data.get('days', []):
                if 'day_of_week' not in day:
                    day['day_of_week'] = 'sabado'
                    
                for i, activity in enumerate(day.get('activities', [])):
                    if 'type' not in activity:
                        activity['type'] = _infer_activity_type(activity.get('location', ''))
                    if 'popularity' not in activity:
                        activity['popularity'] = 7.0
                    if 'estimated_duration_hours' not in activity:
                        activity['estimated_duration_hours'] = _get_default_duration(activity.get('type', 'otro'))
                    if 'distance_from_previous_km' not in activity:
                        activity['distance_from_previous_km'] = 3.0 if i == 0 else 2.0
                    if 'time' not in activity or not activity['time']:
                        # Asignar horarios secuenciales si no existen
                        base_hours = [9, 11, 14, 16, 18, 20]
                        activity['time'] = f"{base_hours[i % len(base_hours)]:02d}:00"
            
            return simulation_data
        return {}
    except Exception as e:
        print(f"Error usando Gemini para estructurar simulación: {e}")
        return {}

def _infer_activity_type(location_name: str) -> str:
    """Infiere el tipo de actividad basado en el nombre del lugar"""
    location_lower = location_name.lower()
    
    type_keywords = {
        'museo': ['museo', 'galería', 'exposición'],
        'restaurante': ['restaurante', 'café', 'cafetería', 'comida', 'almuerzo', 'cena'],
        'parque': ['parque', 'jardín', 'plaza'],
        'monumento': ['monumento', 'estatua', 'memorial', 'catedral', 'iglesia'],
        'playa': ['playa', 'costa', 'mar'],
        'hotel': ['hotel', 'hospedaje', 'alojamiento'],
        'centro_comercial': ['centro comercial', 'shopping', 'tiendas'],
        'teatro': ['teatro', 'cine', 'espectáculo'],
        'zoo': ['zoológico', 'zoo', 'acuario']
    }
    
    for activity_type, keywords in type_keywords.items():
        if any(keyword in location_lower for keyword in keywords):
            return activity_type
    
    return 'otro'

def _get_default_duration(activity_type: str) -> float:
    """Obtiene la duración por defecto según el tipo de actividad"""
    durations = {
        'museo': 1.5,
        'restaurante': 1.2,
        'parque': 1.0,
        'monumento': 0.5,
        'hotel': 0.2,
        'playa': 2.0,
        'centro_comercial': 1.5,
        'teatro': 2.0,
        'zoo': 2.5,
        'otro': 1.0
    }
    return durations.get(activity_type, 1.0)
