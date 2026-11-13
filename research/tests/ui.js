const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1500, height: 1000 }, deviceScaleFactor: 1 });
  const errs = []; p.on('pageerror', e => errs.push('pageerror: ' + e.message)); p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()) });
  await p.goto('http://localhost:8760/'); await p.waitForTimeout(1500);
  await p.screenshot({ path: '/home/user/shots/1_vault.png' });
  for (const k of ['1','2','3','4']) { await p.keyboard.press(k); await p.waitForTimeout(120) }
  await p.waitForTimeout(900); await p.screenshot({ path: '/home/user/shots/1b_wrong.png' });
  for (const k of ['2','4','6','8']) { await p.keyboard.press(k); await p.waitForTimeout(120) }
  await p.waitForTimeout(3500);
  await p.screenshot({ path: '/home/user/shots/2_live_top.png' });
  await p.screenshot({ path: '/home/user/shots/3_live_full.png', fullPage: true });
  await p.click('nav.tabs button[data-v=chart]'); await p.waitForTimeout(2500); await p.screenshot({ path: '/home/user/shots/4_chart.png' });
  await p.click('nav.tabs button[data-v=backtest]'); await p.fill('#btDays', '730'); await p.click('#btRun');
  await p.waitForFunction(() => document.querySelector('#btMsg').textContent.startsWith('done') || document.querySelector('#btMsg').textContent.startsWith('error'), null, { timeout: 180000 });
  await p.waitForTimeout(1500); await p.screenshot({ path: '/home/user/shots/5_backtest.png', fullPage: true });
  console.log('backtest msg:', await p.textContent('#btMsg'));
  await p.click('nav.tabs button[data-v=optimizer]'); await p.fill('#opDays', '730'); await p.fill('#opCombos', '12'); await p.click('#opRun');
  await p.waitForFunction(() => document.querySelector('#opMsg').textContent.startsWith('done') || document.querySelector('#opMsg').textContent.startsWith('error'), null, { timeout: 600000 });
  await p.waitForTimeout(800); await p.screenshot({ path: '/home/user/shots/6_optimizer.png', fullPage: true });
  console.log('opt msg:', await p.textContent('#opMsg'));
  await p.click('nav.tabs button[data-v=control]'); await p.waitForTimeout(1500); await p.screenshot({ path: '/home/user/shots/7_control.png', fullPage: true });
  await p.click('#modes button[data-m=live]'); await p.waitForTimeout(600); await p.screenshot({ path: '/home/user/shots/8_livemode.png' });
  await p.setViewportSize({ width: 390, height: 844 }); await p.click('nav.tabs button[data-v=live]'); await p.waitForTimeout(1500); await p.screenshot({ path: '/home/user/shots/9_mobile.png', fullPage: false });
  const ov = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth); console.log('mobile horizontal overflow px:', ov);
  await p.click('#lockBtn'); await p.waitForTimeout(1200); await p.screenshot({ path: '/home/user/shots/10_locked_mobile.png' });
  console.log('errors:', JSON.stringify(errs.slice(0, 10)));
  await b.close();
})();
