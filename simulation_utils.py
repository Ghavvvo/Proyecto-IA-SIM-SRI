import re
from typing import Dict, Any
import google.generativeai as genai

def format_as_simulation_input(raw_response: str, preferences: dict) -> Dict[str, Any]:
    """
    Uses Gemini to transform a formatted itinerary into a structured format optimal for tourist simulation.
    Only includes information relevant for simulation (days, activities, times, locations, transitions).
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Eres un experto en simulación de turistas. Recibe el siguiente itinerario de viaje y extrae SOLO la información relevante para simular a un turista siguiendo el itinerario.

        Itinerario:
        {raw_response}

        Preferencias del viajero:
        {preferences}

        INSTRUCCIONES:
        - Devuelve un JSON estructurado con la siguiente forma:
          {{
            "destination": string,
            "days": [
              {{
                "day" : int (número del día, empezando desde 1),
                "title": string (titulo del dia ),
                "activities": [
                  {{
                    "time": string (puede estar vacío si no hay hora),
                    "location": string (nombre del lugar o actividad),
                    "description": string (descripcion breve del lugar o actividad),
                  }}
                ]
              }}
            ],
            "interests": list (si está disponible)
          }}
        - NO incluyas narrativa, consejos, ni resúmenes.
        - Solo actividades, lugares, horarios y comidas relevantes para la simulación.
        - Si no hay días, pon todo en un solo día.
        - Si no hay intereses, omite el campo.
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
            return json.loads(json_str)
        return {}
    except Exception as e:
        print(f"Error usando Gemini para estructurar simulación: {e}")
        return {}
