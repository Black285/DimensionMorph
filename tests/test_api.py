"""End-to-end checks over the HTTP API."""

from __future__ import annotations

import json

import pytest

from dimensionmorph import create_app, sources, transforms


@pytest.fixture(scope='module')
def client():
    return create_app(TESTING=True).test_client()


def request_for(source, **overrides):
    body = {
        'source': source.key,
        'expressions': {f.name: f.default for f in source.expressions},
        'ranges': {r.name: {'low': r.low, 'high': r.high} for r in source.ranges},
        'samples': source.samples,
        'parameters': {},
        'views': list(transforms.DEFAULT_VIEWS[source.domain]),
    }
    body.update(overrides)
    return body


def test_index_and_catalog(client):
    assert client.get('/').status_code == 200

    catalog = client.get('/api/catalog').get_json()
    assert {s['key'] for s in catalog['sources']} == {s.key for s in sources.SOURCES}
    assert set(catalog['transforms']) == {'curve', 'surface', 'field'}
    assert catalog['functions']


@pytest.mark.parametrize('source', sources.SOURCES, ids=lambda s: s.key)
def test_every_source_renders_every_view(client, source):
    views = [t.key for t in transforms.for_domain(source.domain)]
    for view in views:
        response = client.post('/api/plot', json=request_for(source, views=[view]))
        assert response.status_code == 200, (source.key, view, response.get_json())
        payload = response.get_json()
        assert payload['ok'], (source.key, view, payload.get('error'))
        assert len(payload['figures']) == 1
        figure = payload['figures'][0]
        assert figure['key'] == view
        assert figure['data'], f'{source.key}/{view} produced no traces'
        assert figure['note']


@pytest.mark.parametrize('source', sources.SOURCES, ids=lambda s: s.key)
def test_responses_are_strict_json(client, source):
    """NaN and Infinity are valid Python json output but JSON.parse rejects them."""
    raw = client.post('/api/plot', json=request_for(source)).get_data(as_text=True)
    assert 'NaN' not in raw
    assert 'Infinity' not in raw
    json.loads(raw)


@pytest.mark.parametrize('source', sources.SOURCES, ids=lambda s: s.key)
def test_every_example_works(client, source):
    for example in source.examples:
        body = request_for(source)
        body['expressions'] = {**body['expressions'], **example['expressions']}
        for name, bounds in example['ranges'].items():
            body['ranges'][name] = bounds
        payload = client.post('/api/plot', json=body).get_json()
        assert payload['ok'], (source.key, example['label'], payload.get('error'))


def test_singularities_do_not_punch_a_hole_at_the_origin(client):
    """Regression: samples near x = 0 used to be discarded for every function."""
    body = request_for(sources.get('explicit2d'), views=['cartesian'], samples=101)
    body['expressions'] = {'y': 'sin(x)'}
    body['ranges'] = {'x': {'low': '-1', 'high': '1'}}
    xs = client.post('/api/plot', json=body).get_json()['figures'][0]['data'][0]['x']
    assert sum(1 for v in xs if v is not None and abs(v) < 0.05) >= 4


def test_a_pole_becomes_a_gap_not_a_spike(client):
    body = request_for(sources.get('explicit2d'), views=['cartesian'], samples=101)
    body['expressions'] = {'y': '1/x'}
    body['ranges'] = {'x': {'low': '-1', 'high': '1'}}
    figure = client.post('/api/plot', json=body).get_json()['figures'][0]
    ys = figure['data'][0]['y']
    assert any(v is None for v in ys), 'the pole should be a null, breaking the line'
    span = figure['layout']['yaxis']['range']
    assert abs(span[1] - span[0]) < 500, 'axis should clip back to the bulk of the data'


def test_reference_grids_are_single_traces(client):
    """The tessellation view used to emit one trace per grid line (196 in total)."""
    body = request_for(sources.get('explicit3d'), views=['triangular'])
    figure = client.post('/api/plot', json=body).get_json()['figures'][0]
    assert len(figure['data']) <= 6


def test_parameters_reach_the_expression(client):
    body = request_for(sources.get('explicit2d'), views=['cartesian'], samples=51)
    body['expressions'] = {'y': 'a*x'}
    body['ranges'] = {'x': {'low': '0', 'high': '1'}}
    body['parameters'] = {'a': 3.0}
    ys = client.post('/api/plot', json=body).get_json()['figures'][0]['data'][0]['y']
    assert max(v for v in ys if v is not None) == pytest.approx(3.0, rel=1e-6)


@pytest.mark.parametrize('body, fragment', [
    ({'source': 'nope', 'expressions': {}}, 'unknown source'),
    ({'source': 'explicit2d', 'expressions': {'y': '().__class__'}}, 'not allowed'),
    ({'source': 'explicit2d', 'expressions': {'y': ''}}, 'missing expression'),
    ({'source': 'explicit2d', 'expressions': {'y': 'sin(x)'},
      'ranges': {'x': {'low': '1', 'high': '1'}}}, 'range is empty'),
    ({'source': 'explicit2d', 'expressions': {'y': 'sin(x)'},
      'parameters': {'x': 1}}, 'reserved'),
    ({'source': 'explicit2d', 'expressions': {'y': 'sin(x)'},
      'parameters': {'a b': 1}}, 'not a valid parameter name'),
    ({'source': 'explicit2d', 'expressions': 'nope'}, 'expressions must be an object'),
])
def test_bad_requests_are_reported_not_raised(client, body, fragment):
    response = client.post('/api/plot', json=body)
    assert response.status_code == 400
    assert fragment in response.get_json()['error']


def test_a_non_object_body_is_rejected(client):
    assert client.post('/api/plot', json=[1, 2, 3]).status_code == 400


def test_unknown_views_fall_back_to_the_defaults(client):
    body = request_for(sources.get('explicit2d'), views=['does-not-exist'])
    payload = client.post('/api/plot', json=body).get_json()
    assert payload['ok']
    assert [f['key'] for f in payload['figures']] == list(transforms.DEFAULT_VIEWS['curve'])


def test_sample_count_is_clamped(client):
    source = sources.get('explicit3d')
    payload = client.post('/api/plot', json=request_for(source, samples=10_000)).get_json()
    assert payload['ok']
    grid = payload['figures'][0]['data'][0]['z']
    assert len(grid) <= source.samples_max


def test_domain_colouring_ships_a_png_not_a_pixel_array(client):
    body = request_for(sources.get('complex'), views=['domain_coloring'])
    trace = client.post('/api/plot', json=body).get_json()['figures'][0]['data'][0]
    assert trace['type'] == 'image'
    assert trace['source'].startswith('data:image/png;base64,')


def test_3d_scenes_start_in_turntable_rotation(client):
    """Plotly seeds scene.dragmode from layout.dragmode, which is 'pan' for the
    flat plots -- every scene has to state turntable for itself."""
    checked = 0
    for source in sources.SOURCES:
        for transform in transforms.for_domain(source.domain):
            if transform.render != 'scene':
                continue
            body = request_for(source, views=[transform.key])
            figure = client.post('/api/plot', json=body).get_json()['figures'][0]
            assert figure['layout']['scene']['dragmode'] == 'turntable', f'{source.key}/{transform.key} would open in pan mode'
            checked += 1
    assert checked >= 8, 'expected several 3D views to check'


def test_flat_plots_keep_pan(client):
    for key, view in (('explicit2d', 'cartesian'), ('complex', 'domain_coloring')):
        body = request_for(sources.get(key), views=[view])
        layout = client.post('/api/plot', json=body).get_json()['figures'][0]['layout']
        assert layout['dragmode'] == 'pan'
        assert 'scene' not in layout


def test_light_and_dark_themes_differ(client):
    source = sources.get('explicit2d')
    light = client.post('/api/plot', json=request_for(source, theme='light')).get_json()
    dark = client.post('/api/plot', json=request_for(source, theme='dark')).get_json()
    assert (light['figures'][0]['layout']['paper_bgcolor'] !=
            dark['figures'][0]['layout']['paper_bgcolor'])
