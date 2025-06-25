from autogen import Agent
import random
import time
import google.generativeai as genai
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from deap import base, creator, tools, algorithms
from typing import List, Dict, Tuple, Callable, Any, Optional
import functools
import json
from utils import format_as_simulation_input

class RouteAgent(Agent):
    def __init__(self, name: str, user_agent: str = "route_optimizer"):
        """
        Agente de optimización de rutas turísticas
        
        Args:
            name: Identificador único del agente
            user_agent: Nombre para el geolocalizador
        """
        super().__init__(name)
        self.geolocator = Nominatim(user_agent=user_agent, timeout=20)
        self.coords_cache = {}
        self.api_counter = 0
        self._setup_deap()
    
    def _setup_deap(self) -> None:
        """Configuración de DEAP"""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
    
    def receive(self, message: Dict, sender: Agent) -> Dict:
        """
        Procesa mensajes recibidos de otros agentes
        
        Args:
            message: Diccionario con el mensaje
            sender: Agente que envió el mensaje
            
        Returns:
            Respuesta en formato dict
        """
        try:
            msg_type = message.get('type')
            
            if msg_type == 'optimize_route':
                return self._handle_optimize_route(message)
                
            elif msg_type == 'generate_itinerary':
                return self._handle_generate_itinerary(message)
                
            elif msg_type == 'get_status':
                return self._get_status()
                
            elif msg_type == 'clear_cache':
                return self._clear_cache()
                
            return {
                'type': 'error',
                'msg': f"Tipo de mensaje no soportado: {msg_type}"
            }
        except Exception as e:
            return {
                'type': 'error',
                'msg': f"Error en {self.name}: {str(e)}"
            }
    
    def _handle_optimize_route(self, message: Dict) -> Dict:
        """Maneja solicitudes de optimización de rutas"""
        places = message["places"]
        params = message.get("parameters", {})
        
        pop_size = params.get("pop_size", 300)
        generations = params.get("generations", 500)
        cx_prob = params.get("cx_prob", 0.7)
        mut_prob = params.get("mut_prob", 0.2)
        
        if len(places) < 2:
            return {
                'type': 'error',
                'msg': "Se necesitan al menos 2 lugares únicos"
            }
        
        start_time = time.time()
        unique_places = list(set(places))
        
        try:
            dist_matrix = self._create_distance_matrix(unique_places)
        except Exception as e:
            return {
                'type': 'error',
                'msg': f"Error al crear matriz de distancias: {str(e)}"
            }
        
        toolbox = base.Toolbox()
        n = len(unique_places)
        
        toolbox.register("indices", random.sample, range(n), n)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("mate", tools.cxPartialyMatched)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", lambda ind: (self._route_distance(ind, dist_matrix),))
        
        population = toolbox.population(n=pop_size)
        
        population, _ = algorithms.eaSimple(
            population, toolbox,
            cxpb=cx_prob, mutpb=mut_prob,
            ngen=generations,
            verbose=False
        )
        
        best_ind = tools.selBest(population, k=1)[0]
        best_distance = best_ind.fitness.values[0]
        ordered_places = [unique_places[i] for i in best_ind]
        
        return {
            'type': 'route_result',
            'order': ordered_places,
            'total_distance_meters': round(best_distance, 2),
            'total_distance_km': round(best_distance / 1000, 2),
            'execution_time': round(time.time() - start_time, 2),
            'api_calls': self.api_counter
        }
    
    def _get_status(self) -> Dict:
        """Devuelve el estado actual del agente"""
        return {
            'type': 'status',
            'cache_size': len(self.coords_cache),
            'api_calls': self.api_counter,
            'status': 'active'
        }
    
    def _clear_cache(self) -> Dict:
        """Limpia la caché de coordenadas"""
        self.coords_cache = {}
        return {
            'type': 'cache_cleared',
            'msg': 'Cache de coordenadas limpiado'
        }
    
    def _get_coordinates(self, place: str) -> Tuple[float, float]:
        """Obtiene coordenadas con caché y usando Gemini como respaldo"""
        if place in self.coords_cache:
            return self.coords_cache[place]
        
        try:
            location = self.geolocator.geocode(place, exactly_one=True)
            if location:
                coords = (location.latitude, location.longitude)
                self.coords_cache[place] = coords
                self.api_counter += 1
                return coords
        except Exception:
            pass 
        
        try:
            original_name = self._get_original_name_with_gemini(place)
            if original_name and original_name != place:
                location = self.geolocator.geocode(original_name, exactly_one=True)
                if location:
                    coords = (location.latitude, location.longitude)
                    self.coords_cache[place] = coords
                    self.coords_cache[original_name] = coords
                    self.api_counter += 1
                    return coords
        except Exception:
            pass  
        
        raise ValueError(f"Lugar no encontrado: {place}")
    
    def _get_original_name_with_gemini(self, place: str) -> Optional[str]:
        """Obtiene el nombre local original usando Gemini"""
        try:
            prompt = (
                f"Dado el nombre turístico '{place}', ¿cuál es su nombre local oficial? "
                "Responde SOLAMENTE con el nombre exacto en su idioma original, "
                "sin comentarios, comillas ni puntuación adicional."
            )
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
        except Exception:
            return None
    
    def _create_distance_matrix(self, places: List[str]) -> List[List[float]]:
        """Crea matriz de distancias entre todos los lugares"""
        coords = [self._get_coordinates(place) for place in places]
        n = len(places)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                dist = geodesic(coords[i], coords[j]).meters
                matrix[i][j] = dist
                matrix[j][i] = dist
        
        return matrix
    
    @staticmethod
    def _route_distance(route: List[int], matrix: List[List[float]]) -> float:
        """Calcula la distancia total de una ruta"""
        total = 0.0
        for i in range(len(route) - 1):
            total += matrix[route[i]][route[i+1]]
        return total
    
    def _handle_generate_itinerary(self, message: Dict) -> Dict:
        """
        Genera un itinerario completo con rutas optimizadas
        
        Args:
            message: Diccionario con:
                - places: Lista de lugares a visitar
                - preferences: Preferencias del usuario (destino, intereses, duración, etc.)
                - days: Número de días para el itinerario (opcional)
                
        Returns:
            Diccionario con el itinerario generado
        """
        try:
            places = message.get('places', [])
            preferences = message.get('preferences', {})
            days = message.get('days', 1)
            
            if len(places) < 2:
                return {
                    'type': 'error',
                    'msg': "Se necesitan al menos 2 lugares para generar un itinerario"
                }
            
            
            if days == 1 and len(places) > 6:
                days = self._estimate_days_needed(len(places), preferences.get('duration', ''))
            
            
            places_per_day = self._distribute_places_by_days(places, days)
            
            
            optimized_routes = {}
            total_distance = 0
            total_travel_time = 0
            
            for day_num, day_places in enumerate(places_per_day, 1):
                if len(day_places) >= 2:
                    
                    route_result = self._handle_optimize_route({
                        'places': day_places,
                        'parameters': {}
                    })
                    
                    if route_result['type'] == 'route_result':
                        
                        travel_times = self._calculate_travel_times(route_result['order'])
                        
                        
                        day_travel_time = sum(t['time_minutes'] for t in travel_times.values())
                        
                        optimized_routes[f'day_{day_num}'] = {
                            'places': route_result['order'],
                            'distance_km': route_result['total_distance_km'],
                            'coordinates': self._get_places_coordinates(route_result['order']),
                            'travel_times': travel_times,
                            'total_travel_time_min': day_travel_time
                        }
                        total_distance += route_result['total_distance_km']
                        total_travel_time += day_travel_time
                else:
                    
                    optimized_routes[f'day_{day_num}'] = {
                        'places': day_places,
                        'distance_km': 0,
                        'coordinates': self._get_places_coordinates(day_places),
                        'travel_times': {},
                        'total_travel_time_min': 0
                    }
            
            
            formatted_itinerary = self._format_itinerary_with_routes(
                optimized_routes, 
                preferences,
                total_distance
            )
            
            
            simulation_json = format_as_simulation_input(formatted_itinerary, preferences)
            
            return {
                'type': 'itinerary_generated',
                'itinerary': formatted_itinerary,
                'routes': optimized_routes,
                'total_distance_km': round(total_distance, 2),
                'days': days,
                'places_count': len(places),
                'simulation_json': simulation_json
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'msg': f"Error generando itinerario: {str(e)}"
            }
    
    def _estimate_days_needed(self, num_places: int, duration_str: str) -> int:
        """Estima el número de días necesarios basado en la cantidad de lugares"""
        import re
        
        
        if duration_str and duration_str != 'No especificada':
            numbers = re.findall(r'\d+', str(duration_str).lower())
            if numbers:
                return int(numbers[0])
            elif 'semana' in duration_str.lower():
                return 7
            elif 'fin de semana' in duration_str.lower():
                return 2
        
        
        return max(1, (num_places + 2) // 3)
    
    def _distribute_places_by_days(self, places: List[str], days: int) -> List[List[str]]:
        """Distribuye los lugares equitativamente entre los días"""
        if days <= 1:
            return [places]
        
        places_per_day = len(places) // days
        remainder = len(places) % days
        
        distribution = []
        start_idx = 0
        
        for day in range(days):
            end_idx = start_idx + places_per_day + (1 if day < remainder else 0)
            distribution.append(places[start_idx:end_idx])
            start_idx = end_idx
        
        return distribution
    
    def _get_places_coordinates(self, places: List[str]) -> List[Dict[str, float]]:
        """Obtiene las coordenadas de una lista de lugares"""
        coordinates = []
        for place in places:
            try:
                lat, lon = self._get_coordinates(place)
                coordinates.append({
                    'place': place,
                    'latitude': lat,
                    'longitude': lon
                })
            except:
                coordinates.append({
                    'place': place,
                    'latitude': None,
                    'longitude': None
                })
        return coordinates
    
    def _calculate_travel_times(self, places: List[str]) -> Dict[str, float]:
        """
        Calcula los tiempos de viaje entre lugares consecutivos usando coordenadas reales
        
        Args:
            places: Lista ordenada de lugares
            
        Returns:
            Diccionario con tiempos de viaje en minutos entre lugares consecutivos
        """
        travel_times = {}
        
        for i in range(len(places) - 1):
            origin = places[i]
            destination = places[i + 1]
            key = f"{origin} → {destination}"
            
            try:
                
                origin_coords = self._get_coordinates(origin)
                dest_coords = self._get_coordinates(destination)
                
                
                distance_km = geodesic(origin_coords, dest_coords).kilometers
                
                
                
                if distance_km <= 1.0:
                    
                    time_minutes = (distance_km / 5) * 60
                    travel_times[key] = {
                        'distance_km': round(distance_km, 2),
                        'time_minutes': round(time_minutes),
                        'mode': 'walking'
                    }
                elif distance_km <= 5.0:
                    
                    time_minutes = (distance_km / 20) * 60
                    travel_times[key] = {
                        'distance_km': round(distance_km, 2),
                        'time_minutes': round(time_minutes),
                        'mode': 'public_transport'
                    }
                else:
                    
                    time_minutes = (distance_km / 30) * 60
                    travel_times[key] = {
                        'distance_km': round(distance_km, 2),
                        'time_minutes': round(time_minutes),
                        'mode': 'taxi'
                    }
                    
            except Exception as e:
                
                print(f"Error calculando tiempo entre {origin} y {destination}: {e}")
                travel_times[key] = {
                    'distance_km': 2.0,  
                    'time_minutes': 15,   
                    'mode': 'estimated'
                }
        
        return travel_times
    
    def _format_itinerary_with_routes(self, routes: Dict, preferences: Dict, total_distance: float) -> str:
        """
        Formatea el itinerario con las rutas optimizadas usando Gemini
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            
            routes_info = ""
            travel_times_info = ""
            
            for day_key, route_data in sorted(routes.items()):
                day_num = day_key.replace('day_', '')
                places_order = " → ".join(route_data['places'])
                distance = route_data['distance_km']
                total_time = route_data.get('total_travel_time_min', 0)
                
                routes_info += f"\nDía {day_num}: {places_order}"
                routes_info += f"\n  - Distancia total: {distance:.1f} km"
                routes_info += f"\n  - Tiempo total de desplazamiento: {total_time} minutos"
                
                
                travel_times_info += f"\n\nTiempos de desplazamiento Día {day_num}:"
                for segment, time_data in route_data.get('travel_times', {}).items():
                    travel_times_info += f"\n- {segment}:"
                    travel_times_info += f"\n  * Distancia: {time_data['distance_km']} km"
                    travel_times_info += f"\n  * Tiempo: {time_data['time_minutes']} minutos"
                    travel_times_info += f"\n  * Modo: {time_data['mode'].replace('_', ' ')}"
            
            prompt = f"""
            Eres un experto planificador de viajes y guía turístico. Crea un itinerario detallado y atractivo.

            Rutas optimizadas por día con distancias y tiempos calculados:
            {routes_info}

            Tiempos de desplazamiento detallados (calculados con coordenadas reales):
            {travel_times_info}

            Preferencias del viajero:
            - Destino: {preferences.get('destination', 'No especificado')}
            - Intereses: {', '.join(preferences.get('interests', ['turismo general']))}
            - Duración: {preferences.get('duration', 'No especificada')}
            - Presupuesto: {preferences.get('budget', 'No especificado')}

            INSTRUCCIONES IMPORTANTES:
            1. USA EXACTAMENTE el orden de lugares proporcionado en las rutas optimizadas
            2. USA EXACTAMENTE los tiempos de desplazamiento calculados que te proporciono
            3. Para cada desplazamiento, indica el modo de transporte recomendado (walking, public transport, taxi)
            4. Incluye horarios específicos para cada lugar considerando:
               - Los tiempos de desplazamiento reales proporcionados
               - Tiempo de visita apropiado para cada tipo de lugar:
                 * Museos: 1.5-2 horas
                 * Parques grandes: 2-3 horas
                 * Monumentos: 30-45 minutos
                 * Restaurantes: 1-1.5 horas
            5. Añade recomendaciones de restaurantes para almuerzo y cena
            6. Incluye consejos prácticos específicos
            7. Usa emojis para hacer el itinerario más visual
            8. Si los lugares están en diferentes ciudades, sugiere dividir por ciudades

            Formato deseado:
            🌟 ITINERARIO OPTIMIZADO PARA {preferences.get('destination', 'TU DESTINO').upper()}

            📊 RESUMEN:
            - Días totales: {len(routes)}
            - Lugares a visitar: {sum(len(r['places']) for r in routes.values())}
            - Distancia total: {total_distance:.1f} km

            📅 DÍA 1: [Título descriptivo]
            📍 Ruta: [Lugar 1] → [Lugar 2] → [Lugar 3]
            📏 Distancia del día: X.X km

            🕐 9:00-10:30 | [Lugar 1]
            📝 [Descripción breve y qué hacer/ver]
            💡 Consejo: [Tip específico]

            🚶 10:30-10:45 | Desplazamiento (X.X km, 15 min caminando)

            🕑 10:45-12:30 | [Lugar 2]
            📝 [Descripción y actividades]
            💡 Consejo: [Tip específico]

            🍽️ 12:30-14:00 | Almuerzo
            📍 Recomendación: [Restaurante cerca con especialidad]

            [Continuar con el resto del día...]

            🏨 Alojamiento sugerido: [Zona recomendada para hospedarse]

            [Repetir formato para cada día]

            💡 CONSEJOS GENERALES:
            - [Consejo sobre transporte]
            - [Consejo sobre horarios/temporada]
            - [Consejo sobre presupuesto]

            🎯 MENSAJE FINAL:
            [Mensaje motivador personalizado según los intereses]
            """

            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            
            print(f"Error formateando con Gemini: {e}")
            return self._format_itinerary_fallback(routes, preferences, total_distance)
    
    def _format_itinerary_fallback(self, routes: Dict, preferences: Dict, total_distance: float) -> str:
        """Formato de itinerario básico como fallback"""
        itinerary = f"🌟 ITINERARIO PARA {preferences.get('destination', 'TU DESTINO').upper()}\n\n"
        itinerary += f"📊 Resumen:\n"
        itinerary += f"- Días: {len(routes)}\n"
        itinerary += f"- Distancia total: {total_distance:.1f} km\n"
        
        
        total_travel_time = sum(r.get('total_travel_time_min', 0) for r in routes.values())
        itinerary += f"- Tiempo total de desplazamiento: {total_travel_time} minutos ({total_travel_time/60:.1f} horas)\n\n"
        
        for day_key, route_data in sorted(routes.items()):
            day_num = day_key.replace('day_', '')
            itinerary += f"📅 DÍA {day_num}:\n"
            itinerary += f"📍 Ruta: {' → '.join(route_data['places'])}\n"
            itinerary += f"📏 Distancia del día: {route_data['distance_km']:.1f} km\n"
            itinerary += f"⏱️ Tiempo de desplazamiento: {route_data.get('total_travel_time_min', 0)} minutos\n"
            
            
            if route_data.get('travel_times'):
                itinerary += "\n🚶 Desplazamientos:\n"
                for segment, time_data in route_data['travel_times'].items():
                    mode_text = {
                        'walking': '🚶 Caminando',
                        'public_transport': '🚌 Transporte público',
                        'taxi': '🚕 Taxi/Auto',
                        'estimated': '❓ Estimado'
                    }.get(time_data['mode'], time_data['mode'])
                    
                    itinerary += f"  • {segment}\n"
                    itinerary += f"    - Distancia: {time_data['distance_km']} km\n"
                    itinerary += f"    - Tiempo: {time_data['time_minutes']} min\n"
                    itinerary += f"    - Modo: {mode_text}\n"
            
            itinerary += "\n"
        
        if preferences.get('interests'):
            itinerary += f"💡 Basado en tus intereses: {', '.join(preferences['interests'])}\n"
        
        return itinerary