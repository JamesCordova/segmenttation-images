# Level set for segmetation

Este proyecto se encarga de hacer uso de algunas imágenes y probar con el algoritmo de la segmetación de imágenes, probando diferentes parametros

## Ejecución

Para ejecutar el programa se debe tener un entorno virtual o hacerlo de forma global en su máquina, se considera que está utilizando Windows

```bash
python -m venv venv
```

Tras ello podemos activar el entorno de python con el siguiente comando

```bash
.\venv\Scripts\activate
```

Ahora podremos instalar los paquetes necesarios, para ello solo se necesita instalar el contenido del arhcivo de requirements

```bash
pip install -r requirements.txt
```

Ahora para comprobar la instalación podemos ejecutar lo siguiente

```bash
pip freeze
```

Para finalizar podemos ejecutar el programa principal

```bash
python -m lv_set.Main
```
