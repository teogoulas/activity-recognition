from utils.constants import ACTIVITIES_MAP


class Activity:

    def __init__(self, activity_type: str, aerobic_training_effect: float, anaerobic_training_effect: float,
                 avg_hr: float, avg_speed: float, calories: float, distance: float, max_ftp: float, max_hr: float,
                 duration: float, begin_timestamp: int, event_type_id: int, start_time_gmt: int, start_time_local: int,
                 elevation_gain: float, elevation_loss: float, max_run_cadence: float, steps: float,
                 avg_stride_length: float, avg_fractional_cadence: float, max_fractional_cadence: float,
                 elapsed_duration: float, moving_duration: float, min_temperature: float, max_temperature: float,
                 min_elevation: float, max_elevation: float, avg_double_cadence: float, max_double_cadence: float,
                 max_vertical_speed: float, lap_count: int, water_estimated: float, min_respiration_rate: float,
                 max_respiration_rate: float, avg_respiration_rate: float, activity_training_load: float):
        self._activity_type = ACTIVITIES_MAP[activity_type]
        self._aerobic_training_effect = aerobic_training_effect
        self._anaerobic_training_effect = anaerobic_training_effect
        self._avg_hr = avg_hr
        self._avg_speed = avg_speed
        self._calories = calories
        self._distance = distance
        self._max_ftp = max_ftp
        self._max_hr = max_hr
        self._duration = duration
        self._begin_timestamp = begin_timestamp
        self._event_type_id = event_type_id
        self._start_time_gmt = start_time_gmt
        self._start_time_local = start_time_local
        self._elevation_gain = elevation_gain
        self._elevation_loss = elevation_loss
        self._max_run_cadence = max_run_cadence
        self._steps = steps
        self._avg_stride_length = avg_stride_length
        self._avg_fractional_cadence = avg_fractional_cadence
        self._max_fractional_cadence = max_fractional_cadence
        self._elapsed_duration = elapsed_duration
        self._moving_duration = moving_duration
        self._min_temperature = min_temperature
        self._max_temperature = max_temperature
        self._min_elevation = min_elevation
        self._max_elevation = max_elevation
        self._avg_double_cadence = avg_double_cadence
        self._max_double_cadence = max_double_cadence
        self._max_vertical_speed = max_vertical_speed
        self._lap_count = lap_count
        self._water_estimated = water_estimated
        self._min_respiration_rate = min_respiration_rate
        self._max_respiration_rate = max_respiration_rate
        self._avg_respiration_rate = avg_respiration_rate
        self._activity_training_load = activity_training_load

    @property
    def activity_type(self) -> str:
        return self._activity_type

    @property
    def aerobic_training_effect(self) -> float:
        return self._aerobic_training_effect

    @property
    def anaerobic_training_effect(self) -> float:
        return self._anaerobic_training_effect

    @property
    def avg_hr(self) -> float:
        return self._avg_hr

    @property
    def avg_speed(self) -> float:
        return self._avg_speed

    @property
    def calories(self) -> float:
        return self._calories

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def max_ftp(self) -> float:
        return self._max_ftp

    @property
    def max_hr(self) -> float:
        return self._max_hr

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def begin_timestamp(self) -> int:
        return self._begin_timestamp

    @property
    def event_type_id(self) -> int:
        return self._event_type_id

    @property
    def start_time_gmt(self) -> float:
        return self._start_time_gmt

    @property
    def start_time_local(self) -> float:
        return self._start_time_local

    @property
    def elevation_gain(self) -> float:
        return self._elevation_gain

    @property
    def elevation_loss(self) -> float:
        return self._elevation_loss

    @property
    def max_run_cadence(self) -> float:
        return self._max_run_cadence

    @property
    def steps(self) -> float:
        return self._steps

    @property
    def avg_stride_length(self) -> float:
        return self._avg_stride_length

    @property
    def avg_fractional_cadence(self) -> float:
        return self._avg_fractional_cadence

    @property
    def max_fractional_cadence(self) -> float:
        return self._max_fractional_cadence

    @property
    def elapsed_duration(self) -> float:
        return self._elapsed_duration

    @property
    def moving_duration(self) -> float:
        return self._moving_duration

    @property
    def min_temperature(self) -> float:
        return self._min_temperature

    @property
    def max_temperature(self) -> float:
        return self._max_temperature

    @property
    def min_elevation(self) -> float:
        return self._min_elevation

    @property
    def max_elevation(self) -> float:
        return self._max_elevation

    @property
    def avg_double_cadence(self) -> float:
        return self._avg_double_cadence

    @property
    def max_double_cadence(self) -> float:
        return self._max_double_cadence

    @property
    def max_vertical_speed(self) -> float:
        return self._max_vertical_speed

    @property
    def lap_count(self) -> float:
        return self._lap_count

    @property
    def water_estimated(self) -> float:
        return self._water_estimated

    @property
    def min_respiration_rate(self) -> float:
        return self._min_respiration_rate

    @property
    def max_respiration_rate(self) -> float:
        return self._max_respiration_rate

    @property
    def avg_respiration_rate(self) -> float:
        return self._avg_respiration_rate

    @property
    def activity_training_load(self) -> float:
        return self._activity_training_load

    def as_dict(self):
        return {'activity_type': self.activity_type, 'aerobic_training_effect': self.aerobic_training_effect,
                'anaerobic_training_effect': self.anaerobic_training_effect, 'avg_hr': self.avg_hr,
                'avg_speed': self.avg_speed, 'calories': self.calories, 'distance': self.distance,
                'max_ftp': self.max_ftp, 'max_hr': self.max_hr, 'duration': self.duration,
                'begin_timestamp': self.begin_timestamp, 'event_type_id': self.event_type_id,
                'start_time_gmt': self.start_time_gmt, 'start_time_local': self.start_time_local,
                'elevation_gain': self.elevation_gain, 'elevation_loss': self.elevation_loss,
                'max_run_cadence': self.max_run_cadence, 'steps': self.steps,
                'avg_stride_length': self.avg_stride_length, 'avg_fractional_cadence': self.avg_fractional_cadence,
                'max_fractional_cadence': self.max_fractional_cadence, 'elapsed_duration': self.elapsed_duration,
                'moving_duration': self.moving_duration, 'min_temperature': self.min_temperature,
                'max_temperature': self.max_temperature, 'min_elevation': self.min_elevation,
                'max_elevation': self.max_elevation, 'avg_double_cadence': self.avg_double_cadence,
                'max_double_cadence': self.max_double_cadence, 'max_vertical_speed': self.max_vertical_speed,
                'lap_count': self.lap_count, 'water_estimated': self.water_estimated,
                'min_respiration_rate': self.min_respiration_rate, 'max_respiration_rate': self.max_respiration_rate,
                'avg_respiration_rate': self.avg_respiration_rate,
                'activity_training_load': self.activity_training_load
                }
