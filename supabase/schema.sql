-- TerraMind AI — Supabase schema
-- Run in the Supabase SQL editor (free tier is enough).

create extension if not exists "pgcrypto";

create table if not exists public.analyses (
    id              uuid primary key default gen_random_uuid(),
    created_at      timestamptz not null default now(),
    source          text not null check (source in ('upload', 'sample', 'gee', 'hotspot')),
    title           text,
    lat             double precision,
    lon             double precision,
    marz            text,
    engine          text,
    area_ha         double precision,
    risk_score      double precision check (risk_score between 0 and 100),
    risk_level      text check (risk_level in ('Low', 'Moderate', 'High', 'Critical')),
    top_issue       text,
    fractions       jsonb,
    issues          jsonb,
    risk            jsonb,
    recommendations jsonb,
    image_url       text,
    user_id         uuid references auth.users(id) default auth.uid()
);

create index if not exists analyses_created_idx on public.analyses (created_at desc);
create index if not exists analyses_marz_idx on public.analyses (marz);
create index if not exists analyses_issue_idx on public.analyses (top_issue);

-- Field reports from rangers / citizens to validate AI detections
create table if not exists public.field_reports (
    id           uuid primary key default gen_random_uuid(),
    created_at   timestamptz not null default now(),
    analysis_id  uuid references public.analyses(id) on delete cascade,
    lat          double precision,
    lon          double precision,
    issue        text,
    confirmed    boolean,
    notes        text,
    photo_url    text,
    user_id      uuid references auth.users(id) default auth.uid()
);

-- Row Level Security
alter table public.analyses enable row level security;
alter table public.field_reports enable row level security;

-- Everyone can read (public transparency dashboard).
create policy "analyses are readable" on public.analyses for select using (true);
create policy "reports are readable" on public.field_reports for select using (true);

-- Inserts: signed-in users, or the anon key for a demo deployment.
-- For production, remove the anon policy and use Supabase Auth.
create policy "anyone can insert analyses (demo)" on public.analyses for insert with check (true);
create policy "owners delete analyses" on public.analyses for delete using (auth.uid() = user_id or user_id is null);
create policy "anyone can insert reports (demo)" on public.field_reports for insert with check (true);

-- Storage bucket for overlay images
insert into storage.buckets (id, name, public) values ('analysis-images', 'analysis-images', true)
on conflict (id) do nothing;
create policy "public read images" on storage.objects for select using (bucket_id = 'analysis-images');
create policy "upload images (demo)" on storage.objects for insert with check (bucket_id = 'analysis-images');

-- Handy view for the dashboard
create or replace view public.issue_summary as
select marz, top_issue, count(*) as n, round(avg(risk_score)::numeric, 1) as avg_risk
from public.analyses group by marz, top_issue;
