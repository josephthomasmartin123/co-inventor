from pydantic_settings import BaseSettings, SettingsConfigDict

OPENROUTER_DEFAULT_PROVIDER = "anthropic"


def _or_model(name: str, use_openrouter: bool) -> str:
    """If using OpenRouter and model name has no '/', prefix with 'anthropic/'."""
    if use_openrouter and "/" not in name:
        return f"{OPENROUTER_DEFAULT_PROVIDER}/{name}"
    return name


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM Provider ---
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""

    # --- Model selection ---
    # Set DEFAULT_MODEL once to apply to every agent.
    # Override individual agents with the specific vars below.
    #
    # Examples (OpenRouter):
    #   DEFAULT_MODEL=deepseek/deepseek-v4-pro
    #   DEFAULT_MODEL=anthropic/claude-sonnet-4-6
    #   DEFAULT_MODEL=google/gemini-2.0-flash-001
    #
    # Examples (Anthropic direct):
    #   DEFAULT_MODEL=claude-sonnet-4-6
    #
    default_model: str = "claude-sonnet-4-6"

    # Per-agent overrides — leave blank to use DEFAULT_MODEL
    generation_model: str = ""
    reflection_model: str = ""
    ranking_model:    str = ""
    evolution_model:  str = ""
    trigger_model:    str = ""

    # --- Search ---
    tavily_api_key: str = ""
    exa_api_key: str = ""

    # --- Pipeline parameters ---
    elo_rounds: int = 12
    top_k_for_evolution: int = 3

    # --- Storage / Server ---
    db_path: str = "co_inventor.db"
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def use_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_llm_key(self) -> bool:
        return bool(self.openrouter_api_key or self.anthropic_api_key)

    def resolved_model(self, override: str) -> str:
        """Return the model to use: override if set, else default_model.
        Auto-prefixes bare names with 'anthropic/' when using OpenRouter."""
        name = override.strip() if override.strip() else self.default_model
        return _or_model(name, self.use_openrouter)

    @property
    def r_generation_model(self) -> str:
        return self.resolved_model(self.generation_model)

    @property
    def r_reflection_model(self) -> str:
        return self.resolved_model(self.reflection_model)

    @property
    def r_ranking_model(self) -> str:
        return self.resolved_model(self.ranking_model)

    @property
    def r_evolution_model(self) -> str:
        return self.resolved_model(self.evolution_model)

    @property
    def r_trigger_model(self) -> str:
        return self.resolved_model(self.trigger_model)


settings = Settings()
