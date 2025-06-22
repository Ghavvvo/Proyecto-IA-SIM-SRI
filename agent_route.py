from autogen import Agent
import random
import time
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from deap import base, creator, tools, algorithms
from typing import List, Dict, Tuple, Callable, Any, Optional
import functools

from mistral_utils import GenerativeModel


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
            model = GenerativeModel("mistral-large-latest")
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
