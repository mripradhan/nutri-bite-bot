-- Re-enable Row Level Security on all application tables.
--
-- Tables created by raw SQL have RLS off by default, so these tables were reachable by
-- anyone holding the project's anon key — which is public by design and shipped to
-- browsers. `patients` holds clinical values (age, labs, diagnosis flags), so that is a
-- data-exposure risk and is inconsistent with the system's data-protection claims.
--
-- No policies are defined deliberately: with RLS enabled and no policy, anon and
-- authenticated roles can read/write nothing. The Flask backend is unaffected because
-- supabase_client.py authenticates with SUPABASE_SERVICE_ROLE_KEY, and the service role
-- bypasses RLS. No frontend talks to Supabase directly.
--
-- Supersedes 20260307163225_disable_rls.sql and the DISABLE statement at the end of
-- 20260923120000_recipe_adherence.sql.
--
-- If a client ever needs direct (non-service-role) access, add explicit policies here
-- rather than disabling RLS again.

ALTER TABLE patients         ENABLE ROW LEVEL SECURITY;
ALTER TABLE recipes          ENABLE ROW LEVEL SECURITY;
ALTER TABLE recipe_adherence ENABLE ROW LEVEL SECURITY;
