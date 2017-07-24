import Attitude_4
import Helpers

att_model = Attitude_4.Model(Helpers.Configuration(epochs=50), 80, 80)
# att_model.train('/Users/Eric/ML_data/Attitude_4/train_data',
#                 '/Users/Eric/ML_data/Attitude_4/validation_data',
#                 '/Users/Eric/ML_data/Attitude_4/test_data')
# predictions = att_model.predict('/Users/Eric/ML_data/Attitude_4/prediction_data')
att_model.train('./data/train_data',
                './data/validation_data',
                './data/test_data')
predictions = att_model.predict('./data/prediction_data')
print(predictions)
