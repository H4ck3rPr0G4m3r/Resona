use serde::Serialize;

use crate::{Direction, Result};

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub direction: Direction,
    pub is_default: bool,
    pub mix_rate: u32,
    pub mix_channels: u16,
}

/// Enumerate active endpoints. Must be callable from any thread (initializes COM MTA).
pub fn list_devices(direction: Direction) -> Result<Vec<DeviceInfo>> {
    let _ = wasapi::initialize_mta();
    let enumerator = wasapi::DeviceEnumerator::new()?;
    let wdir: wasapi::Direction = direction.into();
    let default_id = enumerator.get_default_device(&wdir).and_then(|d| d.get_id()).ok();
    let collection = enumerator.get_device_collection(&wdir)?;
    let mut out = Vec::new();
    for i in 0..collection.get_nbr_devices()? {
        let dev = collection.get_device_at_index(i)?;
        let id = dev.get_id()?;
        let name = dev.get_friendlyname().unwrap_or_else(|_| id.clone());
        let (mix_rate, mix_channels) = dev
            .get_iaudioclient()
            .and_then(|c| c.get_mixformat())
            .map(|f| (f.get_samplespersec(), f.get_nchannels()))
            .unwrap_or((0, 0));
        out.push(DeviceInfo {
            is_default: default_id.as_deref() == Some(id.as_str()),
            id,
            name,
            direction,
            mix_rate,
            mix_channels,
        });
    }
    out.sort_by(|a, b| b.is_default.cmp(&a.is_default).then(a.name.cmp(&b.name)));
    Ok(out)
}

pub(crate) fn open_device(direction: Direction, id: Option<&str>) -> Result<wasapi::Device> {
    let enumerator = wasapi::DeviceEnumerator::new()?;
    match id {
        Some(id) if !id.is_empty() => enumerator
            .get_device(id)
            .map_err(|_| crate::AudioError::NotFound(id.to_string())),
        _ => Ok(enumerator.get_default_device(&direction.into())?),
    }
}
