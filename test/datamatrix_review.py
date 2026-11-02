"""Generate an offline visual review of Data Matrix failures using real scanner traces.

Build datamatrix_trace first. Generated images and traces stay under test/ and
are not tracked. The viewer works through file:// as well as a local web server.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SCANNER = 'CSQR-D-DataMatrix'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jpeg_without_orientation(data):
    """Preserve compressed pixels, strip APP1 so browsers match IMREAD_IGNORE_ORIENTATION."""
    if data[:2] != b'\xff\xd8':
        raise ValueError('Expected JPEG')
    output = bytearray(data[:2])
    pos = 2
    while pos < len(data):
        start = pos
        if data[pos] != 255:
            raise ValueError('Invalid JPEG marker')
        while data[pos] == 255:
            pos += 1
        marker = data[pos]
        pos += 1
        if marker in (0xda, 0xd9):
            output.extend(data[start:])
            return bytes(output)
        length = int.from_bytes(data[pos:pos + 2], 'big')
        if length < 2 or pos + length > len(data):
            raise ValueError('Invalid JPEG segment')
        if marker != 0xe1:
            output.extend(data[start:pos + length])
        pos += length
    raise ValueError('Missing JPEG scan')


def relative(path, root):
    return os.path.relpath(path, root).replace('\\', '/')


def command(args):
    env = os.environ.copy()
    env['PATH'] = 'C:/msys64/ucrt64/bin;' + env.get('PATH', '')
    return subprocess.check_output([str(x) for x in args], env=env, encoding='utf-8')


def compact_case(name, events, base, root, fast, thorough):
    if not any(e['kind'] == 'pass' for e in events):
        return compact_source_case(name, events, base, root, fast, thorough)
    by_id = {e['id']: e for e in events}
    children = collections.defaultdict(list)
    for e in events:
        children[e['parent']].append(e)

    def ancestors(e):
        while e and e.get('parent'):
            e = by_id[e['parent']]
            yield e

    def nearest(e, kind):
        return next((a for a in ancestors(e) if a['kind'] == kind), None)

    def image_of(e):
        return next((a for a in [e, *ancestors(e)] if a.get('image')), None)

    def descendants(e):
        result = []
        todo = list(children[e['id']])
        while todo:
            a = todo.pop()
            result.append(a)
            todo.extend(children[a['id']])
        return result

    passes = []
    for e in events:
        if e['kind'] != 'pass':
            continue
        within = descendants(e)
        selected = [a for a in within if a['kind'] == 'candidate']
        selected_ids = {a['numbers']['candidate'] for a in selected}
        boxes = []
        for a in within:
            if a['kind'] == 'located':
                n = a['numbers']
                boxes.append({'id': n['candidate'], 'points': a['lines'][0],
                              'selected': n['candidate'] in selected_ids, 'texture': bool(n['texture'])})
        ranks = [{**a['numbers'], 'kind': a['kind']} for a in within
                 if a['kind'] in ('coarse_rank', 'fine_rank')]
        counts = collections.Counter(a['kind'] for a in within)
        passes.append({'id': e['id'], 'label': ('Быстрый' if e['numbers']['patch_limit'] == 3 else 'Расширенный') +
                       ' · ' + str(int(e['numbers']['side'])) + ' px',
                       'image': relative(base / e['image'], root), 'width': e['width'], 'height': e['height'],
                       'numbers': e['numbers'], 'boxes': boxes, 'ranks': ranks,
                       'counts': dict(counts), 'selected': [a['id'] for a in selected]})

    counts = collections.Counter(e['kind'] for e in events)
    errors = collections.Counter(str(int(e['numbers'].get('error', 0))) for e in events
                                 if e['kind'] in ('decode', 'uniform_decode') and not e['numbers'].get('valid'))
    errors_messages = collections.Counter((e['strings'].get('message') or 'Без сообщения') for e in events
                                         if e['kind'] in ('decode', 'uniform_decode') and not e['numbers'].get('valid'))
    dims = collections.Counter((int(e['numbers']['w']), int(e['numbers']['h'])) for e in events if e['kind'] == 'recovery')
    success = [e for e in events if e['kind'] in ('decode', 'uniform_decode') and e['numbers'].get('valid')]
    success_ancestors = {a['id'] for e in success for a in ancestors(e)}
    def successful_branch(e):
        return (e in success or e['id'] in success_ancestors or
                (e['kind'] == 'field_lines' and
                 any(a['id'] in success_ancestors and a['kind'] == 'field' for a in ancestors(e))))

    def coordinates(points):
        return [[round(x, 2) if x is not None else None, round(y, 2) if y is not None else None] for x, y in points]
    views = []
    # The complete trace remains available separately. The interactive view
    # keeps all candidate/boundary variants and the first 36 field hypotheses.
    limits = {'field_lines': 36, 'flow_iteration': 16, 'flow_vectors': 12, 'flow_reference': 12, 'clock_gate': 16,
              'read_points': 24, 'decode': 12, 'hard_grid_count': 16}
    used = collections.Counter()
    kinds = {'candidate', 'boundaries', 'straightened', *limits}
    # Prefer the actual ranked recovery hypotheses to preliminary hard grids.
    ordered = sorted(events, key=lambda e: (not successful_branch(e),
                      e['kind'] == 'read_points' and nearest(e, 'recovery') is None, e['id']))
    for e in ordered:
        k = e['kind']
        if k not in kinds or (k == 'hard_grid_count' and e['numbers']['count']) or (k == 'decode' and not e.get('image')):
            continue
        if used[k] >= limits.get(k, 10000) and not successful_branch(e):
            continue
        img = image_of(e)
        if not img:
            continue
        used[k] += 1
        pas = nearest(e, 'pass')
        candidate = e if k == 'candidate' else nearest(e, 'candidate')
        recovery = nearest(e, 'recovery')
        if recovery and not pas:
            origin = by_id.get(int(recovery['numbers'].get('origin', 0)))
            if origin:
                pas = nearest(origin, 'pass')
                # Recover provenance through the prepared patch's identical
                # image, which is the earlier straightened variant.
                prepare = nearest(origin, 'prepare_patch')
                if prepare:
                    match = next((a for a in events if a['kind'] == 'straightened' and a.get('image') == prepare.get('image')), None)
                    if match:
                        candidate = nearest(match, 'candidate')
        dims_event = e if k in ('read_points', 'field_lines') else next(
            (a for a in ancestors(e) if 'w' in a['numbers'] and 'h' in a['numbers']), None)
        n = dict(e['numbers'])
        if dims_event:
            n.setdefault('w', dims_event['numbers']['w'])
            n.setdefault('h', dims_event['numbers']['h'])
        field = nearest(e, 'field')
        if field:
            n['kernel'] = field['numbers']['kernel']
        if recovery:
            n['quality'] = recovery['numbers']['quality']
        lines = e['lines'] if k != 'candidate' else []  # Candidate polygon is in pass coordinates.
        if k == 'clock_gate':
            lines = [[[20, 20], [20, 235]], [[20, 235], [235, 235]]]
        label = k + ' #' + str(e['id'])
        views.append({'id': e['id'], 'kind': k, 'label': label, 'image': relative(base / img['image'], root),
                      'width': img['width'], 'height': img['height'], 'numbers': n,
                      'strings': e['strings'], 'lines': [coordinates(line) for line in lines], 'points': coordinates(e['points']),
                      'pass': pas['id'] if pas else 0, 'candidate': candidate['id'] if candidate else 0,
                      'valid': successful_branch(e)})
    views.sort(key=lambda e: e['id'])
    checksums = errors.get('2', 0)
    if success:
        stage = 'Восстановлен расширенным режимом'
        last = success[0]
        if nearest(last, 'flow'):
            observed = 'Успешное ECC200-декодирование получено после уточнения сетки оптическим потоком.'
        elif nearest(last, 'recovery'):
            observed = 'Успешное ECC200-декодирование получено на этапе восстановления сетки по тактовым сторонам и полю направлений.'
        else:
            observed = 'Успешное ECC200-декодирование получено на дополнительном кандидате или варианте выпрямления.'
    elif counts['decode'] or counts['uniform_decode']:
        stage = 'Отказ декодирования сетки'
        observed = ('Алгоритм выделил кандидатов и построил сетки модулей, но ни одна проверенная матрица не декодировалась. ' +
                    (f'В {checksums} попытках ZXing вернул Checksum: коррекция ошибок не смогла восстановить кодовые слова.'
                     if checksums else 'В проверенных гипотезах получены ошибки структуры/формата.'))
    elif counts['candidate']:
        stage = 'Не восстановлены тактовые дорожки'
        observed = 'Кандидаты выделены, но допустимая сетка ECC200 не дошла до декодирования. Проверьте рамку, L-стороны и число переходов.'
    else:
        stage = 'Не найден кандидат'
        observed = 'После отбора не осталось кандидата для геометрического восстановления.'
    return {'name': name, 'fast': fast, 'thorough': thorough, 'passes': passes, 'views': views,
            'counts': dict(counts), 'errors': dict(errors), 'error_messages': dict(errors_messages),
            'dimensions': [{'w': w, 'h': h, 'attempts': count} for (w, h), count in dims.most_common()],
            'stage': stage, 'observed': observed, 'raw': relative(base / 'events.jsonl.gz', root),
            'raw_events': len(events), 'shown_views': len(views),
            'warning': 'Checksum относится к проверенной гипотезе сетки. Это не доказательство повреждения печати: причиной может быть неверная геометрия, размер сетки или бинаризация.'}


def compact_source_case(name, events, base, root, fast, thorough):
    by_id = {e['id']: e for e in events}
    def ancestors(e):
        while e.get('parent'):
            e = by_id[e['parent']]
            yield e
    result = events[-1]
    width, height = result['numbers']['width'], result['numbers']['height']
    original = relative(ROOT / 'data/datamatrix/data1' / name, root)
    views = []
    searches = [e for e in events if e['kind'] == 'source_search']
    if not searches:  # Traces made before multiscale localization.
        searches = [{'id': 1, 'numbers': {'side': 1000, 'scale': min(1, 1000 / max(width, height))}}]
    passes = {e['id']: {'id': e['id'], 'label': f"Исходное изображение · поиск {int(e['numbers']['side'])} px",
                         'image': original, 'width': width, 'height': height, 'numbers': e['numbers'],
                         'boxes': [], 'ranks': [], 'counts': {}, 'selected': []} for e in searches}
    for e in events:
        chain = [e, *ancestors(e)]
        search = next((a for a in chain if a['kind'] == 'source_search'), None)
        if search is None:
            recovery = next((a for a in chain if a['kind'] == 'source_recovery'), None)
            side = recovery['numbers']['side'] if recovery else None
            search = next((a for a in searches if a['numbers']['side'] == side), searches[0])
        pas = passes[search['id']]
        pas['counts'][e['kind']] = pas['counts'].get(e['kind'], 0) + 1
        if e['kind'] == 'proposal':
            scale = 1 / search['numbers']['scale']
            pas['boxes'].append({'id': e['id'], 'selected': False, 'texture': False,
                          'points': [[x * scale, y * scale] for line in e['lines'] for x, y in line]})
        candidate = next((a for a in chain if a['kind'] in {'candidate', 'recovery_candidate'}), None)
        if e['kind'] == 'boundaries' and candidate:
            shift = candidate['numbers']
            pas['selected'].append(candidate['id'])
            pas['boxes'].append({'id': candidate['id'], 'selected': True, 'texture': True,
                          'points': [[x + shift['crop_x'], y + shift['crop_y']]
                                     for line in e['lines'] for x, y in line]})
        if e['kind'] not in {'candidate', 'recovery_candidate', 'recovery_rejected', 'boundaries', 'boundary_render', 'clock_estimate',
                              'clock_line', 'field_lines', 'field_nodes', 'rectified',
                              'boundary_refinement', 'charted_lines', 'charted_nodes', 'aligned_nodes'}:
            continue
        img = next((a for a in chain if a.get('image')), None)
        if not img:
            continue
        numbers = dict(e['numbers'])
        grid = next((a for a in chain if a['kind'] in {'grid', 'refinement'}), None)
        if grid:
            numbers.update(grid['numbers'])
        # Rectified-event nodes belong to the SOURCE crop, not its render.
        points = [] if e['kind'] == 'rectified' else e['points']
        views.append({'id': e['id'], 'kind': e['kind'], 'label': e['kind'],
                      'image': relative(base / img['image'], root), 'width': img['width'], 'height': img['height'],
                      'numbers': numbers, 'strings': e['strings'], 'lines': e['lines'], 'points': points,
                      'pass': search['id'], 'candidate': candidate['id'] if candidate else 0,
                      'valid': bool(numbers.get('valid'))})
    counts = collections.Counter(e['kind'] for e in events)
    if thorough['decoded']:
        stage, observed = 'Восстановлен расширенным режимом', 'Расширенный режим получил валидное ECC200-декодирование.'
    elif counts['rectified']:
        stage, observed = 'Отказ декодирования сетки', 'Допущены тактовые размеры и построены сетки, но проверенные изображения не декодировались. Сравните границы, узлы и результат TPS.'
    elif counts['candidate']:
        stage, observed = 'Не восстановлены тактовые дорожки', 'Кандидаты проверены, но тактовая оценка не дала допустимой сетки. Проверьте положение контура и бинаризацию тактовых сторон.'
    else:
        stage, observed = 'Не найден кандидат', 'Ни одна область не прошла отбор. Включите все контуры: показаны предварительные предложения, включая отклонённые.'
    dims = collections.Counter((int(e['numbers']['w']), int(e['numbers']['h'])) for e in events if e['kind'] == 'grid')
    return {'name': name, 'fast': fast, 'thorough': thorough, 'passes': list(passes.values()),
            'views': views, 'counts': dict(counts), 'errors': {}, 'error_messages': {},
            'dimensions': [{'w': w, 'h': h, 'attempts': n} for (w, h), n in dims.items()],
            'stage': stage, 'observed': observed, 'raw': relative(base / 'events.jsonl.gz', root),
            'raw_events': len(events), 'shown_views': len(views), 'source_geometry': True,
            'warning': 'Наложения показывают расширенный режим (--thorough); обычный режим ограничивает работу на кадр. Контуры и узлы — гипотезы сканера. Трасса не содержит подробных ошибок ECC200; отказ сам по себе не доказывает повреждение печати.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/datamatrix_review')
    parser.add_argument('--trace-exe', type=Path, default=ROOT / 'src/cpp/build/bin/datamatrix_trace.exe')
    parser.add_argument('--cli', type=Path, default=ROOT / 'src/cpp/build/bin/datamatrix_cli.exe')
    parser.add_argument('--reuse', action='store_true', help='Reuse matching, completed traces')
    parser.add_argument('--skip-parity', action='store_true', help='Skip full-dataset regular-CLI verification')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fast_path, thorough_path = out / 'benchmark_fast.json', out / 'benchmark_thorough.json'
    import sys
    for report_path, mode in [(fast_path, []), (thorough_path, ['--thorough'])]:
        if not (args.reuse and report_path.exists() and read_json(report_path)['executable_sha256'] == sha(args.cli)):
            print(command([sys.executable, ROOT / 'test/datamatrix_benchmark.py', '--exe', args.cli,
                           '--repeat', '3', '--output', report_path, *mode]), flush=True)
    fast_report, thorough_report = read_json(fast_path), read_json(thorough_path)
    fast, thorough = fast_report['results'][SCANNER], thorough_report['results'][SCANNER]
    names = [name for name, result in fast.items() if not result['decoded']]
    trace_sha = sha(args.trace_exe)
    summaries = []
    for index, name in enumerate(names, 1):
        image = ROOT / 'data/datamatrix/data1' / name
        image_sha = sha(image)
        if image_sha != fast_report['manifest'][name]['sha256']:
            raise RuntimeError('Dataset differs from benchmark: ' + name)
        case_dir = out / 'cases' / Path(name).stem
        trace = case_dir / 'trace'
        trace.mkdir(parents=True, exist_ok=True)
        cache = trace / 'provenance.json'
        expected = {'image_sha256': image_sha, 'trace_executable_sha256': trace_sha}
        if not (args.reuse and cache.exists() and read_json(cache) == expected and (trace / 'events.jsonl.gz').exists()):
            print(f'[{index}/{len(names)}] {name}: tracing...', flush=True)
            print(command([args.trace_exe, image, trace, '--thorough']).strip(), flush=True)
            raw = (trace / 'events.jsonl').read_bytes()
            compressed = gzip.compress(raw, compresslevel=6, mtime=0)
            assert gzip.decompress(compressed) == raw
            (trace / 'events.jsonl.gz').write_bytes(compressed)
            cache.write_text(json.dumps(expected), encoding='utf-8')
        else:
            print(f'[{index}/{len(names)}] {name}: cached', flush=True)
        events = [json.loads(line) for line in gzip.decompress((trace / 'events.jsonl.gz').read_bytes()).decode('utf-8').splitlines()]
        result = events[-1]
        assert result['kind'] == 'result'
        if (bool(result['numbers']['decoded']), result['strings']['text']) != (thorough[name]['decoded'], thorough[name]['text']):
            raise RuntimeError('Trace changes scanner result: ' + name)
        case = compact_case(name, events, trace, out, fast[name], thorough[name])
        # SVG/browser JPEG auto-orientation must not rotate source overlays.
        display_image = case_dir / 'source.jpg'
        display_image.write_bytes(jpeg_without_orientation(image.read_bytes()))
        case['original'] = relative(display_image, out)
        if case.get('source_geometry'):
            for pas in case['passes']:
                pas['image'] = case['original']
        case['width'], case['height'] = result['numbers']['width'], result['numbers']['height']
        case['source_sha256'] = image_sha
        case['notes'] = {} if case.get('source_geometry') else read_json(ROOT / 'tools/datamatrix_review/notes.json').get(name, {})
        # script loading works for local files without browser fetch/CORS exceptions.
        data_js = 'window.reviewCase(' + json.dumps(case, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/') + ');'
        (case_dir / 'data.js').write_text(data_js, encoding='utf-8')
        summaries.append({k: case[k] for k in ['name', 'fast', 'thorough', 'stage', 'notes', 'original']})
    parity = None
    if args.skip_parity and (out / 'manifest.json').exists():
        previous = read_json(out / 'manifest.json')
        if previous.get('cli_sha256') == sha(args.cli):
            parity = previous.get('parity')
    if not args.skip_parity:
        parity = {}
        for mode, expected in [('fast', fast), ('thorough', thorough)]:
            print('Verifying regular CLI on all 114 images: ' + mode, flush=True)
            cmd = [args.cli, ROOT / 'data/datamatrix/data1', '--json'] + (['--thorough'] if mode == 'thorough' else [])
            rows = [json.loads(line) for line in command(cmd).splitlines()]
            got = {r['file']: (r['decoded'], r['text']) for r in rows}
            wanted = {name: (r['decoded'], r['text']) for name, r in expected.items()}
            if got != wanted:
                raise RuntimeError('Regular CLI result differs from checkpoint: ' + mode)
            parity[mode] = {'images': len(rows), 'identical_status_and_text': True}
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'git_head': command(['git', 'rev-parse', 'HEAD']).strip(),
                  'trace_executable_sha256': trace_sha, 'cli_sha256': sha(args.cli),
                  'sources': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in
                              [ROOT / 'src/cpp/DMscanner.cpp', ROOT / 'src/cpp/DMTrace.hpp', ROOT / 'src/cpp/datamatrix_trace.cpp',
                               ROOT / 'src/cpp/DMscanner.hpp', ROOT / 'src/cpp/find_dm.cpp', ROOT / 'src/cpp/find_dm.hpp', ROOT / 'src/cpp/dm_correct.cpp', ROOT / 'src/cpp/dm_decode.cpp', ROOT / 'src/cpp/dm_decode.hpp', ROOT / 'src/cpp/dm_correct.hpp', ROOT / 'src/cpp/scan_result.hpp', ROOT / 'src/cpp/vectorfield_correct.cpp',
                               ROOT / 'src/cpp/vectorfield_correct.hpp', ROOT / 'src/cpp/CMakeLists.txt',
                               ROOT / 'test/datamatrix_review.py', ROOT / 'tools/datamatrix_review/index.html',
                               ROOT / 'tools/datamatrix_review/notes.json']},
                  'benchmark_reports': {relative(p, ROOT): sha(p) for p in [fast_path, thorough_path]},
                  'parity': parity, 'cases': [s['name'] for s in summaries],
                  'note': 'Timing comes from saved benchmark reports, not I/O-heavy diagnostic runs. No independent payload labels supplied.'}
    (out / 'manifest.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'summary.js').write_text('window.REVIEW=' + json.dumps({'cases': summaries, 'provenance': provenance}, ensure_ascii=False).replace('</', '<\\/') + ';', encoding='utf-8')
    shutil.copyfile(ROOT / 'tools/datamatrix_review/index.html', out / 'index.html')
    md = ['# Разбор нераспознанных Data Matrix', '', '[Открыть интерактивную подборку](index.html)', '',
          f'{len(names)} отказов быстрого режима; {sum(not thorough[n]["decoded"] for n in names)} остаются нераспознанными расширенным режимом.', '',
          'Время ниже взято из сохранённого тройного бенчмарка; диагностические запуски не измеряют производительность.', '',
          '| Кадр | Быстрый, мс | Расширенный, мс | Результат | Наблюдение / гипотеза |', '|---|---:|---:|---|---|']
    for s in summaries:
        note = s['notes'].get('hypothesis', s['stage'])
        md.append(f'| [{s["name"]}](index.html#{Path(s["name"]).stem}) | {s["fast"]["median_ms"]:.1f} | {s["thorough"]["median_ms"]:.1f} | '
                  + ('Читает thorough' if s['thorough']['decoded'] else 'Не прочитан') + ' | ' + note + ' |')
    md += ['', 'Причины в колонке — визуальные гипотезы. Точный наблюдаемый этап отказа и ошибки каждой проверенной сетки доступны в HTML и полных трассах.',
           '', '[Происхождение данных и проверка паритета](manifest.json)']
    (out / 'README.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('Ready: ' + str(out / 'index.html'), flush=True)


if __name__ == '__main__':
    main()
