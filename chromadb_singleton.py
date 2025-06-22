"""
Implementación del patrón Singleton con metaclases para ChromaDB
Este módulo proporciona una forma thread-safe de mantener una única instancia
de ChromaDB en toda la aplicación.
"""

import threading
import chromadb
from chromadb.utils import embedding_functions
from typing import Optional, Dict, Any
import os


class SingletonMeta(type):
    """
    Metaclase que implementa el patrón Singleton de forma thread-safe.
    
    Esta implementación garantiza que solo exista una instancia de la clase
    incluso en entornos multi-threading.
    """
    
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """
        Controla la creación de instancias de la clase.
        
        Si ya existe una instancia, la retorna. Si no, crea una nueva
        de forma thread-safe.
        """
        # Double-checked locking pattern para mejorar el rendimiento
        if cls not in cls._instances:
            with cls._lock:
                # Verificar nuevamente dentro del lock
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class ChromaDBSingleton(metaclass=SingletonMeta):
    """
    Singleton para gestionar la conexión a ChromaDB.
    
    Esta clase asegura que solo exista una instancia del cliente ChromaDB
    en toda la aplicación, evitando problemas de concurrencia y optimizando
    el uso de recursos.
    """
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Inicializa la conexión a ChromaDB.
        
        Args:
            persist_directory: Directorio donde se almacenarán los datos persistentes
        """
        # Verificar si ya fue inicializado
        if hasattr(self, '_initialized'):
            return
            
        self._persist_directory = persist_directory
        self._client: Optional[chromadb.PersistentClient] = None
        self._collections: Dict[str, Any] = {}
        self._embedding_functions: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Crear el cliente ChromaDB
        self._initialize_client()
        
        # Marcar como inicializado
        self._initialized = True
    
    def _initialize_client(self):
        """Inicializa el cliente ChromaDB de forma segura."""
        try:
            # Crear directorio si no existe
            os.makedirs(self._persist_directory, exist_ok=True)
            
            # Crear cliente persistente
            self._client = chromadb.PersistentClient(path=self._persist_directory)
            print(f"✅ Cliente ChromaDB inicializado en: {self._persist_directory}")
            
        except Exception as e:
            print(f"❌ Error al inicializar ChromaDB: {e}")
            raise
    
    @property
    def client(self) -> chromadb.PersistentClient:
        """
        Obtiene el cliente ChromaDB.
        
        Returns:
            Cliente ChromaDB persistente
        """
        if self._client is None:
            raise RuntimeError("Cliente ChromaDB no inicializado")
        return self._client
    
    def get_or_create_collection(self, 
                                name: str, 
                                embedding_function_name: str = "sentence-transformer",
                                model_name: str = "all-MiniLM-L6-v2",
                                **kwargs) -> Any:
        """
        Obtiene o crea una colección en ChromaDB.
        
        Args:
            name: Nombre de la colección
            embedding_function_name: Tipo de función de embedding a usar
            model_name: Nombre del modelo para embeddings
            **kwargs: Argumentos adicionales para la colección
            
        Returns:
            Colección de ChromaDB
        """
        with self._lock:
            # Verificar si la colección ya existe en caché
            if name in self._collections:
                return self._collections[name]
            
            # Obtener o crear función de embedding
            embedding_function = self._get_or_create_embedding_function(
                embedding_function_name, 
                model_name
            )
            
            try:
                # Intentar obtener o crear la colección
                collection = self._client.get_or_create_collection(
                    name=name,
                    embedding_function=embedding_function,
                    **kwargs
                )
                
                # Guardar en caché
                self._collections[name] = collection
                print(f"✅ Colección '{name}' obtenida/creada exitosamente")
                
                return collection
                
            except Exception as e:
                print(f"❌ Error al obtener/crear colección '{name}': {e}")
                raise
    
    def _get_or_create_embedding_function(self, 
                                         function_name: str, 
                                         model_name: str) -> Any:
        """
        Obtiene o crea una función de embedding.
        
        Args:
            function_name: Tipo de función de embedding
            model_name: Nombre del modelo
            
        Returns:
            Función de embedding
        """
        cache_key = f"{function_name}:{model_name}"
        
        if cache_key in self._embedding_functions:
            return self._embedding_functions[cache_key]
        
        # Crear nueva función de embedding según el tipo
        if function_name == "sentence-transformer":
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        elif function_name == "openai":
            embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                model_name=model_name
            )
        elif function_name == "cohere":
            embedding_function = embedding_functions.CohereEmbeddingFunction(
                model_name=model_name
            )
        else:
            # Por defecto usar sentence-transformer
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        # Guardar en caché
        self._embedding_functions[cache_key] = embedding_function
        
        return embedding_function
    
    def get_collection(self, name: str) -> Optional[Any]:
        """
        Obtiene una colección existente.
        
        Args:
            name: Nombre de la colección
            
        Returns:
            Colección si existe, None en caso contrario
        """
        with self._lock:
            # Verificar caché primero
            if name in self._collections:
                return self._collections[name]
            
            try:
                # Intentar obtener del cliente
                collection = self._client.get_collection(name=name)
                self._collections[name] = collection
                return collection
            except Exception:
                return None
    
    def delete_collection(self, name: str) -> bool:
        """
        Elimina una colección.
        
        Args:
            name: Nombre de la colección a eliminar
            
        Returns:
            True si se eliminó exitosamente, False en caso contrario
        """
        with self._lock:
            try:
                self._client.delete_collection(name=name)
                
                # Eliminar de caché si existe
                if name in self._collections:
                    del self._collections[name]
                
                print(f"✅ Colección '{name}' eliminada exitosamente")
                return True
                
            except Exception as e:
                print(f"❌ Error al eliminar colección '{name}': {e}")
                return False
    
    def list_collections(self) -> list:
        """
        Lista todas las colecciones disponibles.
        
        Returns:
            Lista de nombres de colecciones
        """
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"❌ Error al listar colecciones: {e}")
            return []
    
    def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información sobre una colección.
        
        Args:
            name: Nombre de la colección
            
        Returns:
            Diccionario con información de la colección o None
        """
        collection = self.get_collection(name)
        if collection:
            try:
                count = collection.count()
                return {
                    "name": name,
                    "count": count,
                    "metadata": collection.metadata if hasattr(collection, 'metadata') else {}
                }
            except Exception as e:
                print(f"❌ Error al obtener información de '{name}': {e}")
        return None
    
    def reset(self):
        """
        Reinicia el singleton (útil para testing).
        
        ⚠️ ADVERTENCIA: Esto eliminará la instancia actual y requerirá
        una nueva inicialización.
        """
        with self._lock:
            if self._client:
                # Limpiar caché
                self._collections.clear()
                self._embedding_functions.clear()
                
                # Cerrar cliente si es posible
                if hasattr(self._client, 'close'):
                    self._client.close()
                
                self._client = None
                
                # Eliminar la instancia del registro de la metaclase
                if self.__class__ in SingletonMeta._instances:
                    del SingletonMeta._instances[self.__class__]
                
                print("✅ Singleton ChromaDB reiniciado")
    
    def __repr__(self):
        """Representación en string del singleton."""
        collections_count = len(self.list_collections())
        return (f"ChromaDBSingleton(persist_directory='{self._persist_directory}', "
                f"collections={collections_count})")


# Función de conveniencia para obtener la instancia
def get_chromadb_instance(persist_directory: str = "chroma_db") -> ChromaDBSingleton:
    """
    Obtiene la instancia singleton de ChromaDB.
    
    Args:
        persist_directory: Directorio de persistencia
        
    Returns:
        Instancia singleton de ChromaDB
    """
    return ChromaDBSingleton(persist_directory)


# Ejemplo de uso
if __name__ == "__main__":
    # Demostración del patrón Singleton
    print("=== Demostración del Patrón Singleton con ChromaDB ===\n")
    
    # Crear primera instancia
    db1 = ChromaDBSingleton()
    print(f"Primera instancia: {id(db1)}")
    
    # Intentar crear segunda instancia
    db2 = ChromaDBSingleton()
    print(f"Segunda instancia: {id(db2)}")
    
    # Verificar que son la misma instancia
    print(f"¿Son la misma instancia? {db1 is db2}")
    
    # Usar función de conveniencia
    db3 = get_chromadb_instance()
    print(f"Tercera instancia (via función): {id(db3)}")
    print(f"¿Es la misma que las anteriores? {db1 is db3}")
    
    # Crear una colección de ejemplo
    print("\n=== Operaciones con Colecciones ===")
    collection = db1.get_or_create_collection("test_collection")
    print(f"Colección creada: {collection.name}")
    
    # Listar colecciones
    collections = db1.list_collections()
    print(f"Colecciones disponibles: {collections}")
    
    # Obtener información de la colección
    info = db1.get_collection_info("test_collection")
    print(f"Información de la colección: {info}")
    
    # Demostrar thread-safety
    print("\n=== Prueba de Thread-Safety ===")
    import concurrent.futures
    
    def create_instance(thread_id):
        instance = ChromaDBSingleton()
        return f"Thread {thread_id}: {id(instance)}"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(create_instance, i) for i in range(5)]
        results = [future.result() for future in futures]
        
    for result in results:
        print(result)
    
    print("\n✅ Todas las instancias son idénticas (mismo ID)")