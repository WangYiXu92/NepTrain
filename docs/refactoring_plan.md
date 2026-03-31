# Perturbation Engine Refactoring Plan

## Current Issues

1. **Redundant Defect Pre-processing**: Every defect is applied to a `dummy_atoms` object just to estimate atom counts for Sobol dimensions
2. **Scattered Dimension Logic**: Dimension calculations are disconnected from the code that consumes them
3. **Manual Sample Slicing**: Fragile dimension alignment that breaks silently when defect order changes
4. **Monolithic Function**: `perturb()` is ~1400 lines handling everything from parsing to filtering

## Proposed Solution: Defect Handler Registry

### Architecture

```python
class DefectHandler:
    """Base class for all defect handlers"""
    def get_dims(self, atoms, **params) -> int:
        """Return number of Sobol dimensions needed"""
        pass
    
    def apply(self, atoms, samples, **params) -> Atoms:
        """Apply perturbation using provided samples"""
        pass
    
    def get_metadata(self, **params) -> dict:
        """Return metadata for annotation"""
        pass

# Example handlers
class SurfaceHandler(DefectHandler):
    def get_dims(self, atoms, surface_vacuum, surface_layers, **params):
        dims = 0
        if _is_range(surface_vacuum): dims += 1
        if _is_range(surface_layers): dims += 1
        return dims
    
    def apply(self, atoms, samples, surface_indices, surface_vacuum, surface_layers, **params):
        # Parse samples
        idx = 0
        vac = surface_vacuum
        lay = surface_layers
        
        if _is_range(surface_vacuum):
            vac = _interpolate(samples[idx], *_parse_range(surface_vacuum))
            idx += 1
        
        if _is_range(surface_layers):
            lay = _discrete_sample(samples[idx], *_parse_range(surface_layers))
            idx += 1
        
        return generate_surface(atoms, indices=surface_indices, vacuum=vac, layers=lay)
```

### Registry System

```python
DEFECT_HANDLERS = {
    'surface': SurfaceHandler(),
    'grain_boundary': GrainBoundaryHandler(),
    'dislocation': DislocationHandler(),
    'twinning': TwinningHandler(),
    'stacking_fault': StackingFaultHandler(),
    'amorphous': AmorphousHandler(),
    'magnetic': MagneticHandler(),
    'rotation': RotationHandler(),
    'vacancy': VacancyHandler(),
    'shuffle': ShuffleHandler(),
    'volume': VolumeHandler(),
}

# Defect execution order
DEFECT_ORDER = [
    'surface', 'grain_boundary', 'dislocation', 'twinning', 
    'stacking_fault', 'amorphous', 'magnetic', 'rotation',
    'vacancy', 'shuffle', 'volume'
]
```

### Refactored `perturb()` Loop

```python
def perturb(atoms, num=1, sampler='sobol', **kwargs):
    # 1. Determine active defects
    active_defects = []
    for defect_name in DEFECT_ORDER:
        if kwargs.get(f'{defect_name}', False):
            active_defects.append(defect_name)
    
    # 2. Calculate total dimensions
    total_dims = 9  # Cell strain
    for defect_name in active_defects:
        handler = DEFECT_HANDLERS[defect_name]
        total_dims += handler.get_dims(atoms, **kwargs)
    
    # 3. Initialize sampler
    s_sampler = SobolSampler(d=total_dims) if sampler == 'sobol' else RandomSampler(d=total_dims)
    
    # 4. Generate structures
    for i in range(num):
        struct = atoms.copy()
        samples = s_sampler.random(n=1)[0]
        
        # Apply cell strain
        cell_samples = samples[:9]
        struct = apply_cell_strain(struct, cell_samples, **kwargs)
        
        # Apply defects in order
        dim_offset = 9
        for defect_name in active_defects:
            handler = DEFECT_HANDLERS[defect_name]
            n_dims = handler.get_dims(struct, **kwargs)
            defect_samples = samples[dim_offset:dim_offset + n_dims]
            struct = handler.apply(struct, defect_samples, **kwargs)
            dim_offset += n_dims
        
        # Filtering
        if validate_structure and not adjust_reasonable(struct):
            continue
        
        yield struct
```

## Benefits

1. **No Redundancy**: Dimension calculation and application are in the same handler
2. **Type Safety**: Each handler knows its own parameter types
3. **Easy Extension**: Add new defects by creating a new handler class
4. **Testable**: Each handler can be unit-tested independently
5. **Maintainable**: ~1400 line function becomes ~200 lines + handlers

## Implementation Steps

1. Create `DefectHandler` base class
2. Implement handlers for each defect type (11 total)
3. Create registry and execution order
4. Refactor `perturb()` to use registry
5. Update tests to verify backward compatibility
6. Remove old dimension calculation code

## Timeline

- Phase 1 (Handlers): 2-3 hours
- Phase 2 (Integration): 1-2 hours  
- Phase 3 (Testing): 1 hour
- **Total**: 4-6 hours of focused work
