# Identity Effect

The identity effect returns its input unchanged. It serves as the foundational primitive in the effects registry.

## Purpose

- **Testing**: Verify the effects pipeline is working correctly
- **No-op placeholder**: Use when an effect slot requires a value but no transformation is needed
- **Composition base**: The neutral element for effect composition

## Signature

```
identity(input) → input
```

## Properties

- **Idempotent**: `identity(identity(x)) = identity(x)`
- **Neutral**: For any effect `f`, `identity ∘ f = f ∘ identity = f`

## Implementation

```python
def identity(input):
    return input
```

## Content Hash

The identity effect is content-addressed by its behavior: given any input, the output hash equals the input hash.

## Owner

Registered by `@giles@artdag.rose-ash.com`
