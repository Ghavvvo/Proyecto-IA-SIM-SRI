"""
Configuración centralizada para Google Gemini AI
Este módulo proporciona una interfaz unificada para todas las interacciones con Gemini
"""

import google.generativeai as genai
import os
from typing import Optional, Dict, List, Union, Any
from functools import lru_cache
import json
import re
from datetime import datetime
import threading
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class GeminiConfig:
    """
    Clase singleton para gestionar la configuración de Gemini
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Obtener API key
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            # Intentar con la key hardcodeada como fallback (no recomendado en producción)
            self.api_key = "AIzaSyCuiFY0aCJEaOndmd_jEHZIabbA23TWn6E"
        
        # Configurar Gemini
        genai.configure(api_key=self.api_key)
        
        # Modelos disponibles
        self.models = {
            'flash': 'gemini-1.5-flash',
            'pro': 'gemini-1.5-pro',
            'flash-8b': 'gemini-1.5-flash-8b'
        }
        
        # Modelo por defecto
        self.default_model = 'flash'
        
        # Configuraciones de generación por defecto
        self.default_generation_config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # Cache de modelos
        self._model_cache = {}
        
        # Estadísticas de uso
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'requests_by_model': {},
            'errors': []
        }
        
        self._initialized = True
        print("✅ Gemini configurado correctamente")
    
    def get_model(self, model_name: str = None, **generation_config) -> genai.GenerativeModel:
        """
        Obtiene una instancia del modelo Gemini
        
        Args:
            model_name: Nombre del modelo ('flash', 'pro', 'flash-8b') o nombre completo
            **generation_config: Configuración de generación personalizada
            
        Returns:
            Instancia del modelo GenerativeModel
        """
        # Usar modelo por defecto si no se especifica
        if model_name is None:
            model_name = self.default_model
        
        # Convertir alias a nombre completo
        if model_name in self.models:
            model_name = self.models[model_name]
        
        # Crear clave de cache
        config_key = json.dumps(generation_config, sort_keys=True) if generation_config else 'default'
        cache_key = f"{model_name}:{config_key}"
        
        # Verificar cache
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Combinar configuración por defecto con personalizada
        final_config = self.default_generation_config.copy()
        final_config.update(generation_config)
        
        # Crear modelo
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=final_config
        )
        
        # Guardar en cache
        self._model_cache[cache_key] = model
        
        return model
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de uso"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reinicia las estadísticas"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'requests_by_model': {},
            'errors': []
        }


class GeminiClient:
    """
    Cliente principal para interactuar con Gemini
    """
    
    def __init__(self, model_name: str = None, **generation_config):
        """
        Inicializa el cliente Gemini
        
        Args:
            model_name: Nombre del modelo a usar
            **generation_config: Configuración de generación
        """
        self.config = GeminiConfig()
        self.model_name = model_name or self.config.default_model
        self.generation_config = generation_config
        self.model = self.config.get_model(self.model_name, **generation_config)
    
    def generate(self, 
                prompt: str, 
                system_instruction: str = None,
                response_format: str = "text",
                max_retries: int = 3,
                **kwargs) -> Union[str, Dict, None]:
        """
        Genera una respuesta usando Gemini
        
        Args:
            prompt: Prompt para el modelo
            system_instruction: Instrucción del sistema (opcional)
            response_format: Formato de respuesta esperado ('text', 'json', 'structured')
            max_retries: Número máximo de reintentos en caso de error
            **kwargs: Argumentos adicionales para generate_content
            
        Returns:
            Respuesta generada en el formato especificado
        """
        # Actualizar estadísticas
        self.config.stats['total_requests'] += 1
        model_key = self.model_name
        if model_key not in self.config.stats['requests_by_model']:
            self.config.stats['requests_by_model'][model_key] = 0
        self.config.stats['requests_by_model'][model_key] += 1
        
        # Preparar el prompt completo
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"
        
        # Intentar generar respuesta con reintentos
        last_error = None
        for attempt in range(max_retries):
            try:
                # Generar respuesta
                response = self.model.generate_content(full_prompt, **kwargs)
                
                # Procesar según formato esperado
                if response_format == "json":
                    result = self._parse_json_response(response.text)
                elif response_format == "structured":
                    result = self._parse_structured_response(response.text)
                else:
                    result = response.text.strip()
                
                # Actualizar estadísticas de éxito
                self.config.stats['successful_requests'] += 1
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"⚠️ Error en intento {attempt + 1}/{max_retries}: {str(e)}")
                    continue
        
        # Si llegamos aquí, todos los intentos fallaron
        self.config.stats['failed_requests'] += 1
        self.config.stats['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': str(last_error),
            'prompt_preview': prompt[:100] + '...' if len(prompt) > 100 else prompt
        })
        
        print(f"❌ Error después de {max_retries} intentos: {str(last_error)}")
        return None
    
    def generate_json(self, prompt: str, schema: Dict = None, **kwargs) -> Optional[Dict]:
        """
        Genera una respuesta en formato JSON
        
        Args:
            prompt: Prompt para el modelo
            schema: Esquema JSON esperado (opcional)
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con la respuesta o None si falla
        """
        # Añadir instrucciones para formato JSON
        json_prompt = prompt
        if schema:
            json_prompt += f"\n\nDevuelve la respuesta en formato JSON siguiendo este esquema:\n{json.dumps(schema, indent=2)}"
        else:
            json_prompt += "\n\nDevuelve la respuesta ÚNICAMENTE en formato JSON válido, sin explicaciones adicionales."
        
        return self.generate(json_prompt, response_format="json", **kwargs)
    
    def generate_structured(self, 
                          prompt: str, 
                          structure_template: str,
                          **kwargs) -> Optional[Dict]:
        """
        Genera una respuesta siguiendo una estructura específica
        
        Args:
            prompt: Prompt para el modelo
            structure_template: Plantilla de la estructura esperada
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con la respuesta estructurada
        """
        structured_prompt = f"{prompt}\n\nSigue esta estructura:\n{structure_template}"
        return self.generate(structured_prompt, response_format="structured", **kwargs)
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """
        Parsea una respuesta JSON de Gemini
        
        Args:
            response_text: Texto de respuesta del modelo
            
        Returns:
            Diccionario parseado o None si falla
        """
        try:
            # Intentar parsear directamente
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Intentar extraer JSON del texto
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Intentar con limpieza adicional
            cleaned = response_text.strip()
            # Eliminar comillas de código si existen
            cleaned = re.sub(r'^```json\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"⚠️ No se pudo parsear JSON de la respuesta")
                return None
    
    def _parse_structured_response(self, response_text: str) -> Dict:
        """
        Parsea una respuesta estructurada de Gemini
        
        Args:
            response_text: Texto de respuesta del modelo
            
        Returns:
            Diccionario con la información estructurada
        """
        # Intentar primero como JSON
        json_result = self._parse_json_response(response_text)
        if json_result:
            return json_result
        
        # Si no es JSON, parsear como texto estructurado
        result = {}
        lines = response_text.strip().split('\n')
        
        current_key = None
        current_value = []
        
        for line in lines:
            # Detectar claves (líneas que terminan en ':')
            if ':' in line and not line.strip().startswith('-'):
                if current_key:
                    result[current_key] = '\n'.join(current_value).strip()
                
                parts = line.split(':', 1)
                current_key = parts[0].strip().lower().replace(' ', '_')
                current_value = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
            elif current_key:
                current_value.append(line.strip())
        
        # Guardar último par clave-valor
        if current_key:
            result[current_key] = '\n'.join(current_value).strip()
        
        return result
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Interfaz de chat con historial de mensajes
        
        Args:
            messages: Lista de mensajes con formato [{"role": "user/assistant", "content": "..."}]
            **kwargs: Argumentos adicionales
            
        Returns:
            Respuesta del asistente
        """
        # Construir prompt desde el historial
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                prompt_parts.append(f"Usuario: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Asistente: {content}")
            elif role == 'system':
                prompt_parts.append(f"Sistema: {content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Añadir indicador para la respuesta
        full_prompt += "\n\nAsistente:"
        
        response = self.generate(full_prompt, **kwargs)
        
        return response.strip() if response else None


# Funciones de conveniencia
@lru_cache(maxsize=1)
def get_gemini_client(model_name: str = None, **kwargs) -> GeminiClient:
    """
    Obtiene una instancia del cliente Gemini (con cache)
    
    Args:
        model_name: Nombre del modelo
        **kwargs: Configuración adicional
        
    Returns:
        Cliente Gemini configurado
    """
    return GeminiClient(model_name, **kwargs)


def gemini_generate(prompt: str, 
                   model: str = "flash",
                   response_format: str = "text",
                   **kwargs) -> Union[str, Dict, None]:
    """
    Función rápida para generar respuestas con Gemini
    
    Args:
        prompt: Prompt para el modelo
        model: Modelo a usar ('flash', 'pro', 'flash-8b')
        response_format: Formato de respuesta ('text', 'json', 'structured')
        **kwargs: Argumentos adicionales
        
    Returns:
        Respuesta generada
    """
    client = get_gemini_client(model)
    return client.generate(prompt, response_format=response_format, **kwargs)


def gemini_json(prompt: str, schema: Dict = None, model: str = "flash", **kwargs) -> Optional[Dict]:
    """
    Función rápida para generar respuestas JSON con Gemini
    
    Args:
        prompt: Prompt para el modelo
        schema: Esquema JSON esperado
        model: Modelo a usar
        **kwargs: Argumentos adicionales
        
    Returns:
        Diccionario con la respuesta
    """
    client = get_gemini_client(model)
    return client.generate_json(prompt, schema, **kwargs)


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    print("=== Prueba de Configuración Centralizada de Gemini ===\n")
    
    # Prueba 1: Generación de texto simple
    print("1. Generación de texto simple:")
    response = gemini_generate("¿Cuál es la capital de Francia?")
    print(f"Respuesta: {response}\n")
    
    # Prueba 2: Generación de JSON
    print("2. Generación de JSON:")
    json_response = gemini_json(
        "Dame información sobre París en formato JSON con campos: nombre, pais, poblacion, atracciones",
        schema={
            "nombre": "string",
            "pais": "string", 
            "poblacion": "number",
            "atracciones": ["string"]
        }
    )
    print(f"Respuesta JSON: {json.dumps(json_response, indent=2, ensure_ascii=False)}\n")
    
    # Prueba 3: Cliente personalizado
    print("3. Cliente personalizado con configuración específica:")
    custom_client = GeminiClient(
        model_name="flash",
        temperature=0.9,
        max_output_tokens=500
    )
    creative_response = custom_client.generate(
        "Escribe un haiku sobre programación",
        system_instruction="Eres un poeta experto en haikus"
    )
    print(f"Haiku:\n{creative_response}\n")
    
    # Prueba 4: Estadísticas
    print("4. Estadísticas de uso:")
    config = GeminiConfig()
    stats = config.get_stats()
    print(f"Total de solicitudes: {stats['total_requests']}")
    print(f"Solicitudes exitosas: {stats['successful_requests']}")
    print(f"Solicitudes fallidas: {stats['failed_requests']}")
    print(f"Solicitudes por modelo: {stats['requests_by_model']}")