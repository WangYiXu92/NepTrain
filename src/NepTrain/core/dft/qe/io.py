import os
import re
from pathlib import Path
from ase.io.espresso import read_espresso_in

def get_pp_files(directory="."):
    """
    Find pseudopotential files (.UPF) in the directory.
    Returns a dictionary {Element: filename}.
    """
    pp_files = {}
    path = Path(directory)
    # Search for UPF files (case insensitive extension usually .UPF or .upf)
    upfs = list(path.glob("*.UPF")) + list(path.glob("*.upf"))
    
    for upf in upfs:
        try:
            # Try to guess element from filename (common naming convention Element.pbe...)
            # or read the file content
            filename = upf.name
            # Heuristic: split by dot or underscore or hyphen, first part is element
            # This is fragile, so let's try reading the file header
            with open(upf, "r", encoding="utf8", errors='ignore') as f:
                header = f.read(2000) # Read first 2000 chars
            
            # Look for element="Mg" or element: Mg
            match = re.search(r'(?i)(?:element\s*=\s*|element:\s*|AtomType\s*=\s*)"?([A-Za-z]{1,2})"?', header)
            if match:
                elem = match.group(1).capitalize()
                pp_files[elem] = filename
            else:
                # Fallback to filename
                parts = re.split(r'[._-]', filename)
                possible_elem = parts[0].capitalize()
                if len(possible_elem) <= 2:
                     pp_files[possible_elem] = filename
        except Exception:
            pass
            
    return pp_files

def read_qe_input(file_path):
    """
    Read a QE input file and return parameters and pseudopotentials.
    Uses ASE's read_espresso_in but handles errors if structure is missing.
    """
    if not os.path.exists(file_path):
        return {}, {}
        
    try:
        # Try reading as a full structure file
        with open(file_path, 'r') as f:
            atoms = read_espresso_in(f)
        
        input_data = atoms.calc.parameters
        pseudopotentials = atoms.calc.pseudopotentials
        return input_data, pseudopotentials
    except Exception:
        # Fallback: simple parsing if ASE fails (e.g. no atoms)
        # This is a simplified parser for namelists
        input_data = {}
        pseudopotentials = {}
        
        current_namelist = None
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
                
            if line.startswith('&'):
                current_namelist = line[1:].upper()
                # input_data[current_namelist] = {} 
                # ASE expects flat dict for input_data usually, or sectioned? 
                # ASE Espresso calculator expects 'control', 'system', 'electrons' keys in input_data dict
                # or a flat dict? ASE documentation says input_data is a dict.
                # Let's flatten it but keep keys unique? 
                # Actually ASE puts them into sections.
                continue
            
            if line == '/':
                current_namelist = None
                continue
                
            if current_namelist:
                # Parse key = value
                # The line might contain multiple key=value pairs if they are comma separated?
                # ASE format usually puts one per line, but user might not.
                # The error showed "ibrav= 2, celldm(1) =10.20, ..." being parsed as value for ibrav.
                # My simple split('=', 1) takes everything after first = as value.
                
                # To fix this, we need a better parser for comma-separated key-values on one line.
                # Or just split by comma first?
                # But comma might be inside quotes? (unlikely for numbers)
                
                # Split line by comma first?
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if not part: continue
                    if '=' in part:
                        key, val = part.split('=', 1)
                        key = key.strip().lower()
                        val = val.strip().strip('"').strip("'")
                        
                        # Try to convert to number or bool
                        if val.lower() == '.true.':
                            val = True
                        elif val.lower() == '.false.':
                            val = False
                        else:
                            try:
                                if '.' in val:
                                    val = float(val)
                                else:
                                    val = int(val)
                            except:
                                pass
                        
                        # Store in input_data
                        if current_namelist.lower() not in input_data:
                            input_data[current_namelist.lower()] = {}
                        input_data[current_namelist.lower()][key] = val

            # Handle ATOMIC_SPECIES for pseudopotentials if parsing manually
            # This is complex, better rely on user providing PPs or auto-discovery
            
        return input_data, pseudopotentials
