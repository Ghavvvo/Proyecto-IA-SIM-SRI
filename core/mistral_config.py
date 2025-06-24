"""
Configuración centralizada para Mistral AI
Este módulo proporciona una interfaz unificada para todas las interacciones con Mistral
"""

from mistralai import Mistral
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


class MistralConfig:
    """
    Clase singleton para gestionar la configuración de Mistral
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
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY no encontrada en las variables de entorno")
        
        # Configurar cliente Mistral
        self.client = Mistral(api_key=self.api_key)
        
        # Modelos disponibles
        self.models = {
            'flash': 'mistral-small-latest',  # Equivalente a Gemini Flash
            'pro': 'mistral-large-latest',    # Equivalente a Gemini Pro
            'flash-8b': 'mistral-small-latest'  # Usando small como equivalente
        }
        
        # Modelo por defecto
        self.default_model = 'flash'
        
        # Configuraciones de generación por defecto
        self.default_generation_config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'max_tokens': 2048,
        }
        
        # Cache de configuraciones
        self._config_cache = {}
        
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
        print("✅ Mistral configurado correctamente")
    
    def get_model_name(self, model_name: str = None) -> str:
        """
        Obtiene el nombre completo del modelo
        
        Args:
            model_name: Nombre del modelo ('flash', 'pro', 'flash-8b') o nombre completo
            
        Returns:
            Nombre completo del modelo
        """
        # Usar modelo por defecto si no se especifica
        if model_name is None:
            model_name = self.default_model
        
        # Convertir alias a nombre completo
        if model_name in self.models:
            return self.models[model_name]
        
        return model_name
    
    def get_generation_config(self, **kwargs) -> Dict[str, Any]:
        """
        Obtiene la configuración de generación combinada
        
        Args:
            **kwargs: Configuración personalizada
            
        Returns:
            Configuración final
        """
        # Combinar configuración por defecto con personalizada
        final_config = self.default_generation_config.copy()
        final_config.update(kwargs)
        return final_config
    
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


class MistralClient:
    """
    Cliente principal para interactuar con Mistral
    """
    
    def __init__(self, model_name: str = None, **generation_config):
        """
        Inicializa el cliente Mistral
        
        Args:
            model_name: Nombre del modelo a usar
            **generation_config: Configuración de generación
        """
        self.config = MistralConfig()
        self.model_name = model_name or self.config.default_model
        self.generation_config = generation_config
        self.model = self.config.get_model_name(self.model_name)
    
    def generate(self, 
                prompt: str, 
                system_instruction: str = None,
                response_format: str = "text",
                max_retries: int = 3,
                **kwargs) -> Union[str, Dict, None]:
        """
        Genera una respuesta usando Mistral
        
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
        
        # Preparar mensajes
        messages = []
        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction
            })
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Obtener configuración final
        final_config = self.config.get_generation_config(**self.generation_config, **kwargs)
        
        # Intentar generar respuesta con reintentos
        last_error = None
        for attempt in range(max_retries):
            try:
                # Generar respuesta usando la API de Mistral
                response = self.config.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=final_config.get('temperature', 0.7),
                    top_p=final_config.get('top_p', 0.95),
                    max_tokens=final_config.get('max_tokens', 2048),
                )
                
                # Extraer el contenido de la respuesta
                response_text = response.choices[0].message.content
                
                # Procesar según formato esperado
                if response_format == "json":
                    result = self._parse_json_response(response_text)
                elif response_format == "structured":
                    result = self._parse_structured_response(response_text)
                else:
                    result = response_text.strip()
                
                # Actualizar estadísticas de éxito
                self.config.stats['successful_requests'] += 1
                if hasattr(response, 'usage'):
                    self.config.stats['total_tokens'] += response.usage.total_tokens
                
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
        Parsea una respuesta JSON de Mistral
        
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
        Parsea una respuesta estructurada de Mistral
        
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
        # Convertir mensajes al formato de Mistral
        mistral_messages = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Mistral usa 'system', 'user', 'assistant'
            if role in ['system', 'user', 'assistant']:
                mistral_messages.append({
                    "role": role,
                    "content": content
                })
        
        # Obtener configuración final
        final_config = self.config.get_generation_config(**self.generation_config, **kwargs)
        
        try:
            # Generar respuesta
            response = self.config.client.chat.complete(
                model=self.model,
                messages=mistral_messages,
                temperature=final_config.get('temperature', 0.7),
                top_p=final_config.get('top_p', 0.95),
                max_tokens=final_config.get('max_tokens', 2048),
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error en chat: {str(e)}")
            return None


# Funciones de conveniencia (mantienen la misma interfaz que gemini_config.py)
@lru_cache(maxsize=1)
def get_mistral_client(model_name: str = None, **kwargs) -> MistralClient:
    """
    Obtiene una instancia del cliente Mistral (con cache)
    
    Args:
        model_name: Nombre del modelo
        **kwargs: Configuración adicional
        
    Returns:
        Cliente Mistral configurado
    """
    return MistralClient(model_name, **kwargs)


def mistral_generate(prompt: str, 
                   model: str = "flash",
                   response_format: str = "text",
                   **kwargs) -> Union[str, Dict, None]:
    """
    Función rápida para generar respuestas con Mistral
    
    Args:
        prompt: Prompt para el modelo
        model: Modelo a usar ('flash', 'pro', 'flash-8b')
        response_format: Formato de respuesta ('text', 'json', 'structured')
        **kwargs: Argumentos adicionales
        
    Returns:
        Respuesta generada
    """
    client = get_mistral_client(model)
    return client.generate(prompt, response_format=response_format, **kwargs)


def mistral_json(prompt: str, schema: Dict = None, model: str = "flash", **kwargs) -> Optional[Dict]:
    """
    Función rápida para generar respuestas JSON con Mistral
    
    Args:
        prompt: Prompt para el modelo
        schema: Esquema JSON esperado
        model: Modelo a usar
        **kwargs: Argumentos adicionales
        
    Returns:
        Diccionario con la respuesta
    """
    client = get_mistral_client(model)
    return client.generate_json(prompt, schema, **kwargs)


# Alias para mantener compatibilidad con la interfaz anterior
GeminiClient = MistralClient
GeminiConfig = MistralConfig
gemini_generate = mistral_generate
gemini_json = mistral_json
get_gemini_client = get_mistral_client


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    print("=== Prueba de Configuración Centralizada de Mistral ===\n")
    
    # Prueba 1: Generación de texto simple
    print("1. Generación de texto simple:")
    response = mistral_generate("¿Cuál es la capital de Francia?")
    print(f"Respuesta: {response}\n")
    
    # Prueba 2: Generación de JSON
    print("2. Generación de JSON:")
    json_response = mistral_json(
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
    custom_client = MistralClient(
        model_name="flash",
        temperature=0.9,
        max_tokens=500
    )
    creative_response = custom_client.generate(
        "Escribe un haiku sobre programación",
        system_instruction="Eres un poeta experto en haikus"
    )
    print(f"Haiku:\n{creative_response}\n")
    
    # Prueba 4: Estadísticas
    print("4. Estadísticas de uso:")
    config = MistralConfig()
    stats = config.get_stats()
    print(f"Total de solicitudes: {stats['total_requests']}")
    print(f"Solicitudes exitosas: {stats['successful_requests']}")
    print(f"Solicitudes fallidas: {stats['failed_requests']}")
    print(f"Solicitudes por modelo: {stats['requests_by_model']}")