# implement_q_exit.py
"""
Implementa correctamente la funcionalidad de salir con 'q'
"""

# Leer el archivo
with open('src/main_bot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("🔧 Implementando salida con 'q'...")

# Buscar el bucle principal del menú
modified = False
for i in range(len(lines)):
    line = lines[i]
    
    # Buscar donde se ejecuta la opción seleccionada
    if 'opciones[opcion]()' in line or 'actions[choice]()' in line:
        print(f"✅ Encontrado ejecutor de opciones en línea {i+1}")
        
        # Buscar el input que sigue después
        for j in range(i+1, min(i+10, len(lines))):
            if 'input(' in lines[j] and ('Enter' in lines[j] or 'menú' in lines[j]):
                print(f"✅ Encontrado input en línea {j+1}")
                
                # Obtener la indentación
                indent = len(lines[j]) - len(lines[j].lstrip())
                spaces = ' ' * indent
                
                # Reemplazar con código que maneja 'q'
                if '= input(' not in lines[j]:
                    # No tiene asignación, agregar
                    new_lines = [
                        f'{spaces}user_response = {lines[j].strip()}\n',
                        f'{spaces}if user_response.lower() in ["q", "9"]:\n',
                        f'{spaces}    print("\\n👋 ¡Hasta luego!")\n',
                        f'{spaces}    break\n'
                    ]
                    lines[j:j+1] = new_lines
                    modified = True
                    print(f"✅ Código de salida agregado")
                break

# Si no encontramos el patrón anterior, buscar de otra forma
if not modified:
    print("\n🔍 Buscando patrón alternativo...")
    
    # Buscar todos los inputs con "q para salir"
    for i in range(len(lines)):
        if 'input(' in lines[i] and 'q para salir' in lines[i]:
            # Ver si está dentro de una lambda
            if 'lambda:' in lines[i] or (i > 0 and 'lambda:' in lines[i-1]):
                print(f"⚠️  Input en lambda en línea {i+1} - no se puede modificar directamente")
                continue
            
            # Si no está en lambda, podemos modificarlo
            indent = len(lines[i]) - len(lines[i].lstrip())
            spaces = ' ' * indent
            
            # Si no tiene asignación
            if '= input(' not in lines[i]:
                input_line = lines[i].strip()
                new_lines = [
                    f'{spaces}user_response = {input_line}\n',
                    f'{spaces}if user_response.lower() in ["q", "9"]:\n',
                    f'{spaces}    break\n'
                ]
                lines[i:i+1] = new_lines
                modified = True
                print(f"✅ Modificada línea {i+1}")

# Guardar si hubo modificaciones
if modified:
    with open('src/main_bot.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("\n✅ Archivo guardado con funcionalidad de salida")
else:
    print("\n⚠️  No se pudieron hacer modificaciones automáticas")
    print("\n📝 Alternativa: Crear una función wrapper...")
    
    # Agregar una función helper al inicio del archivo
    helper_function = '''
def safe_input(prompt="Presiona Enter para continuar..."):
    """Input que permite salir con q o 9"""
    response = input(prompt)
    if response.lower() in ['q', '9', 'quit', 'exit']:
        print("\\n👋 ¡Hasta luego!")
        import sys
        sys.exit(0)
    return response

'''
    
    # Insertar después de los imports
    import_end = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('import') and not line.startswith('from'):
            import_end = i
            break
    
    lines.insert(import_end, helper_function)
    
    # Reemplazar los input() con safe_input()
    for i in range(len(lines)):
        if 'input(' in lines[i] and 'q para salir' in lines[i]:
            lines[i] = lines[i].replace('input(', 'safe_input(')
    
    # Guardar
    with open('src/main_bot.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("✅ Agregada función safe_input() y reemplazados los inputs")

print("\n📌 Ahora puedes salir con:")
print("   - 'q' o 'Q'")
print("   - '9'")
print("   - 'quit' o 'exit'")
print("\n✅ Ejecuta: python src/main_bot.py")