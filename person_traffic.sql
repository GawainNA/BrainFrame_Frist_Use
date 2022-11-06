SELECT zone_status.tstamp as time, 
       total_count.count_enter AS entered, 
      total_count.count_exit AS exited FROM zone_status
LEFT JOIN total_count ON total_count.zone_status_id=zone_status.id
LEFT JOIN zone ON zone_status.zone_id=zone.id
WHERE
  total_count.class_name='person' AND
  zone.name='Nearby' AND
  zone_status.id >= (SELECT id FROM zone_status WHERE tstamp >= $__unixEpochFrom() ORDER BY tstamp ASC LIMIT 1) AND
  zone_status.id <= (SELECT id FROM zone_status WHERE tstamp < $__unixEpochTo() ORDER BY tstamp DESC LIMIT 1) 