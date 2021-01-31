class Activity:

    def __init__(self, activity_type: str, aerobic_training_effect: float, anaerobic_training_effect: float,
                 avg_hr: float, avg_speed: float,
                 calories: float, distance: float,  # max_double_cadence: float,
                 max_ftp: float, max_hr: float, duration: float):
        self._activity_type = activity_type
        self._aerobic_training_effect = aerobic_training_effect
        self._anaerobic_training_effect = anaerobic_training_effect
        self._avg_hr = avg_hr
        self._avg_speed = avg_speed
        self._calories = calories
        self._distance = distance if distance is not None else 0.0
        # self._max_double_cadence = max_double_cadence
        self._max_ftp = max_ftp
        self._max_hr = max_hr
        self._duration = duration

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

    # @property
    # def max_double_cadence(self) -> float:
    #     return self._max_double_cadence

    @property
    def max_ftp(self) -> float:
        return self._max_ftp

    @property
    def max_hr(self) -> float:
        return self._max_hr

    @property
    def duration(self) -> float:
        return self._duration

    def as_dict(self):
        return {'activity_type': self.activity_type, 'aerobic_training_effect': self.aerobic_training_effect,
                'anaerobic_training_effect': self.anaerobic_training_effect, 'avg_hr': self.avg_hr,
                'avg_speed': self.avg_speed, 'calories': self.calories, 'distance': self.distance,
                # 'max_double_cadence': self.max_double_cadence,
                'max_ftp': self.max_ftp, 'max_hr': self.max_hr, 'duration': self.duration}
