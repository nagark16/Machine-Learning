def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    input_feature_sum = input_feature.sum()
    output_sum = output.sum()
    
    # compute the product of the output and the input_feature and its sum
    temp = (input_feature*output).sum()
    
    # compute the squared value of the input_feature and its sum
    temp2 = (input_feature*input_feature).sum()
        
    # use the formula for the slope
    num = temp - ((input_feature_sum*output_sum)/(len(input_feature)))
    den = temp2 - ((input_feature_sum*input_feature_sum)/(len(input_feature)))
    slope = num/den
    
    # use the formula for the intercept
    intercept = (output_sum/len(input_feature)) - (slope * (input_feature_sum/len(input_feature)))
    
    return (intercept, slope)

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + (slope * input_feature)
    return predicted_values

#evaluate above simple regression model using Residual Sum of Squares (RSS)
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predictions = get_regression_predictions(input_feature, intercept, slope)

    #if we have access to model instead of intercept & slope
    #predictions = model.predict(data)

    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residual = output - predictions

    # square the residuals and add them up
    RSS = (residual * residual).sum()

    return(RSS)

def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output-intercept)/slope

    return estimated_feature

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]

    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]

    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)

def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix,weights)

    return(predictions)

def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2 * np.dot(errors,feature)

    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix,weights)

        # compute the errors as predictions - output
        errors = predictions - output

        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors,feature_matrix[:,i])

            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += derivative * derivative

            # subtract the step size times the derivative from the current weight
            weights[i] -= (step_size * derivative)
            
        # compute the square-root of the gradient sum of squares to get the gradient magnitude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature

    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature ** power

    return poly_sframe

def build_model(data):
    poly2_data = polynomial_sframe(data['sqft_living'], 15)
    my_features = poly2_data.column_names() # get the name of the features
    poly2_data['price'] = data['price'] # add price to the data since it's the target
    model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
    return poly2_data,model2
    
def coeff(data):
    new_data, model = build_model(data)
    print model.get("coefficients")
    return new_data, model
    
def graph(data):
    new_data, model = coeff(data)
    plt.plot(new_data['power_1'],new_data['price'],'.',
            new_data['power_1'], model.predict(new_data),'-')

training_and_validation, testing = sales.random_split(0.9,seed=1)
training,validation = training_and_validation.random_split(0.5,seed=1)

finalRss = []
for i in range(1,15+1):
    poly2_data = polynomial_sframe(training['sqft_living'], i)
    my_features = poly2_data.column_names() # get the name of the features
    poly2_data['price'] = training['price'] # add price to the data since it's the target
    model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None,verbose=False)
    validation_poly = polynomial_sframe(validation['sqft_living'],i)
    predictions = model2.predict(validation_poly)
    rss = validation['price'] - predictions
    rss = (rss*rss).sum()
    finalRss.append(rss)

print(finalRss.index(min(finalRss)), min(finalRss))
    
def polynomial_features(data, deg):
    data_copy=data.copy()
    for i in range(1,deg):
        data_copy['X'+str(i+1)]=data_copy['X'+str(i)]*data_copy['X1']
    return data_copy

def polynomial_regression(data, deg):
    model = graphlab.linear_regression.create(polynomial_features(data,deg), 
                                              target='Y', l2_penalty=0.,l1_penalty=0.,
                                              validation_set=None,verbose=False)
    return model

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature

    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature ** power

    return poly_sframe

#cross validation
(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)


def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    rss_sum = 0
    n = len(data)
    for i in range(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation_set = data[start:end+1]
        training_set = data[0:start].append(data[end+1:n])
        model = graphlab.linear_regression.create(training_set, target = output_name, 
                                           features = features_list,l2_penalty=l2_penalty, validation_set = None,verbose=False)
        predictions = model.predict(validation_set)
        residuals = validation_set['price'] - predictions
        rss = sum(residuals*residuals)
        rss_sum += rss
    validation_error = rss_sum/k
    return validation_error
        
    
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix,axis=0)
    normalized_features = feature_matrix/norms
    return (normalized_features,norms)

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix,weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    temp1 = weights[i]*feature_matrix[:,i]
    temp2 = output - prediction
    ro_i = sum(feature_matrix[:,i]*(temp2 +temp1))

    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = (ro_i+l1_penalty/2.)
    elif ro_i > l1_penalty/2.:
        new_weight_i = (ro_i-l1_penalty/2.)
    else:
        new_weight_i = 0.
    
    return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    D = feature_matrix.shape[1]
    change = np.array(initial_weights) * 0.0
    weights = np.array(initial_weights)
    converged = False
    
    while not converged:
        for iter in range(D):
            weight = lasso_coordinate_descent_step(iter, feature_matrix, output, weights, l1_penalty)
            change[iter] = np.abs(weight - weights[iter])
            weights[iter] = weight
        max_change = max(change)
        if max_change < tolerance:
            converged = True
    return weights
    
(feature_matrix, output_array) = get_numpy_data(train_data, all_features, 'price')
(normalized_features,norms) = normalize_features(feature_matrix)
normalized_weigths1e7 = weights1e7/norms
normalized_weigths1e8 = weights1e8/norms
normalized_weigths1e4 = weights1e4/norms
print(normalized_weigths1e7[3])

def checkDist(featureMatrix,query):
    diff = featureMatrix - query
    distances = np.sqrt(np.sum(diff**2,axis=1))
    return distances

def knn(k, featureMatrix, query):
    distances = checkDist(featureMatrix, query)
    return np.argsort(distances,axis=0)[:k]

def avgKNN(k, featureMatrix, output,query):
    k_neigbors = knn(k,featureMatrix,query)
    avg_value = np.mean(output[k_neigbors])
    return avg_value


#bias -- how well can our model fit the true relationship averaging over all possible training sets we might see